import gc
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Literal

import h5py
import numpy as np
import torch
import wandb

from src.matching.matcher import Matcher
from src.matching.retriever import Retriever
from src.matching.sparse.keypoint_detector import KeypointDetector, KeypointDetectorCfg
from src.matching.sparse.keypoint_matcher import KeypointMatcher, KeypointMatcherCfg
from src.matching.tracking.tracker import Tracker, TrackerCfg
from src.matching.tracking.trajectory import TrajectorySet

mp.set_start_method("spawn", force=True)


@dataclass
class MatcherTrackingCfg:
    name: Literal["tracking"]
    tracker: TrackerCfg
    keypoint_detector: KeypointDetectorCfg
    keypoint_matcher: KeypointMatcherCfg


class MatcherTracking(Matcher[MatcherTrackingCfg]):
    def __init__(
        self,
        cfg: MatcherTrackingCfg,
        logger: wandb.sdk.wandb_run.Run,
        device: torch.device,
        retriever: Retriever,
    ):
        super().__init__(cfg, logger, device, retriever)

        self.tracker = Tracker(cfg.tracker, logger)
        self.detector = KeypointDetector(cfg.keypoint_detector, logger, device)
        self.matcher = KeypointMatcher(cfg.keypoint_matcher, logger)

    def match(self, image_paths: list[Path], feature_dir: Path) -> None:
        """Track points over frames in dynamic cameras and match keypoints in a fixed camera."""
        if (feature_dir / "matches.h5").exists():
            print("Matches in the fixed camera already exist, skipping tracking.")
            return

        start = time()

        print("Track points over frames in dynamic cameras...")
        self.tracker.track(image_paths, feature_dir)
        lap_tracking = time()
        print(f"Tracking completed in {(lap_tracking - start) // 60:.2f} minutes.")

        torch.cuda.empty_cache()
        gc.collect()

        print("Merge trajectories from all dynamic cameras...")
        track_path = feature_dir / "track.npy"
        if track_path.exists():
            print("\t Loading pre-merged trajectories...")
            trajectories = np.load(track_path, allow_pickle=True).item()
        else:
            print("\t Merging trajectories from all cameras...")
            full_trajs = []
            for cam_name in ["2_dynA", "3_dynB", "4_dynC"]:
                trajs = np.load(
                    feature_dir / cam_name / "full_trajs.npy", allow_pickle=True
                )
                full_trajs.extend(trajs)
            # Build TrajectorySet
            dict_trajs = {}
            for idx, traj in enumerate(full_trajs):
                if traj.length() < self.cfg.tracker.traj_min_len:
                    continue
                dict_trajs[idx] = traj
            trajectories = TrajectorySet(dict_trajs)
            trajectories.build_invert_indexes()
            np.save(track_path, trajectories, allow_pickle=True)

        print("Register keypoints in all cameras...")
        if (feature_dir / "keypoints.h5").exists():
            print("\t Keypoints already registered, skipping keypoint detection.")
        else:
            print("\t Register keypoints in a fixed camera...")
            self.detector.detect_keypoints(
                image_paths[: len(image_paths) // 4],
                feature_dir=feature_dir,
            )
            with h5py.File(
                feature_dir / "keypoints.h5", mode="a"
            ) as f_keypoints, h5py.File(
                feature_dir / "descriptors.h5", mode="a"
            ) as f_descriptors:
                print("\t Register keypoints in dynamic cameras...")
                for frame_id, trajs_dict in sorted(trajectories.invert_maps.items()):
                    key = "-".join(image_paths[frame_id].parts[-3:])
                    kp_list = []
                    desc_list = []
                    for traj_id, idx_in_traj in sorted(trajs_dict.items()):
                        traj = trajectories.trajs[traj_id]
                        kp_list.append(traj.xys[idx_in_traj])
                        desc_list.append(traj.descs[idx_in_traj])
                    f_keypoints[key] = np.stack(kp_list, dtype=np.float32)
                    f_descriptors[key] = np.stack(desc_list, dtype=np.float32)

        torch.cuda.empty_cache()
        gc.collect()

        print("Match keypoints in the fixed camera exhaustively.")
        if (feature_dir / "matches.h5").exists():
            print(
                "\t Matches already registered, skipping matching in the fixed camera."
            )
        else:
            index_pairs = self.retriever.get_index_pairs(
                image_paths, "fixed", self.cfg.tracker.window_len
            )
            self.matcher.match_keypoints_fixed(index_pairs, image_paths, feature_dir)

        torch.cuda.empty_cache()
        gc.collect()

        # print("Match keypoints in dynamic cameras exhaustively.")
        # index_pairs = self.retriever.get_index_pairs(
        #     image_paths, "exhaustive_keyframe", self.cfg.tracker.window_len
        # )
        # self.matcher.match_keypoints_traj(
        #     image_paths, trajectories, index_pairs, feature_dir
        # )

        # # Get keypoints for each image from trajectories
        # trajectories.build_invert_indexes()
        # keypoints_per_image = {}
        # for frame_id, trajs_dict in trajectories.invert_maps.items():
        #     kp_list = []
        #     desc_list = []
        #     traj_list = []
        #     idx_in_traj_list = []
        #     # NOTE: Make sure invert_maps[frame_id] are sorted by traj_id
        #     for traj_id, idx_in_traj in sorted(trajs_dict.items()):
        #         traj = trajectories.trajs[traj_id]
        #         kp = traj.xys[idx_in_traj]  # (x, y) location at this frame
        #         desc = traj.descs[idx_in_traj]
        #         kp_list.append(kp)
        #         desc_list.append(desc)
        #         traj_list.append(traj_id)
        #         idx_in_traj_list.append(idx_in_traj)  # Store index in trajectory
        #     keypoints_per_image[int(frame_id)] = (
        #         np.array(kp_list),
        #         np.array(desc_list),
        #         np.array(traj_list, dtype=int),
        #         np.array(idx_in_traj_list, dtype=int),
        #     )

        # index_pairs = self.retriever.get_index_pairs(
        #     image_paths, "exhaustive_keyframe", self.cfg.tracker.window_len
        # )
        # # Match trajectories for each pair: dict[frame_i][frame_j] -> (traj_id_i, traj_id_j)
        # matched_traj_ids = self.matcher.match_keypoints_traj(
        #     image_paths, keypoints_per_image, index_pairs
        # )

        # torch.cuda.empty_cache()
        # gc.collect()

        # # Propagate matches to trajectories
        # for frame_i in tqdm(matched_traj_ids):
        #     for frame_j in matched_traj_ids[frame_i]:
        #         traj_pairs = matched_traj_ids[frame_i][
        #             frame_j
        #         ]  # List of (traj_id_i, traj_id_j)
        #         for traj_id_i, traj_id_j in zip(traj_pairs[0], traj_pairs[1]):
        #             traj_j = trajectories.trajs[traj_id_j]
        #             # Append data from traj_j to traj_i
        #             trajectories.trajs[traj_id_i].times.extend(traj_j.times)
        #             trajectories.trajs[traj_id_i].xys.extend(traj_j.xys)
        #             # trajectories.trajs[traj_id_i].descs.extend(traj_j.descs)

        # with h5py.File(feature_dir / "keypoints.h5", mode="w") as f_keypoints:
        #     # Get keypoints for each image from trajectories
        #     trajectories.build_invert_indexes()
        #     kpts_indices = {}
        #     for frame_id, trajs_dict in trajectories.invert_maps.items():
        #         # key = "-".join(image_paths[frame_id].parts[-3:])
        #         kp_list = []
        #         kpts_indices_map = {}
        #         # NOTE: Make sure invert_maps[frame_id] are sorted by traj_id
        #         for traj_id, idx_in_traj in sorted(trajs_dict.items()):
        #             traj = trajectories.trajs[traj_id]
        #             kp = traj.xys[idx_in_traj]  # (x, y) location at this frame
        #             kp_list.append(kp + 0.5)
        #             kpts_indices_map[traj_id] = len(kp_list) - 1
        #         kpts_indices[int(frame_id)] = kpts_indices_map
        #         f_keypoints[str(frame_id)] = np.array(kp_list)

        # trajectories.build_invert_indexes()
        # mask_dir = Path(str(image_paths[0].parent).replace("images", "masks"))
        # mask_imgs = [
        #     cv2.imread(mask_dir / pth.name, cv2.IMREAD_GRAYSCALE) for pth in image_paths
        # ]
        # trajectories.build_match_indexes(
        #     feature_dir, mask_imgs, kpts_indices, len(image_paths) // 4, i_proc=0
        # )

        end = time()

        self.logger.log({"matching_time": (end - start) // 60})
        self.logger.summary["tracking_time"] = (lap_tracking - start) // 60
        self.logger.summary["sparse_time"] = (end - lap_tracking) // 60
