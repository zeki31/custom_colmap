import multiprocessing as mp

mp.set_start_method("spawn", force=True)

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Literal

import h5py
import numpy as np
import torch
import wandb
from tqdm import tqdm
import cv2
from src.matching.matcher import Matcher
from src.matching.retriever import Retriever
from src.matching.sparse.keypoint_detector import KeypointDetector, KeypointDetectorCfg
from src.matching.sparse.keypoint_matcher import KeypointMatcher, KeypointMatcherCfg
from src.matching.tracking.tracker import Tracker, TrackerCfg
from src.matching.tracking.trajectory import TrajectorySet
import gc

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
        if (feature_dir / "matches_0.npy").exists():
            print("Matching already done, skipping.")
            return

        start = time()
        futures = []
        with ProcessPoolExecutor(max_workers=2) as executor:
            for i_proc, cam_name in enumerate(
                ["1_fixed", "2_dynA", "3_dynB", "4_dynC"]
            ):
                sub_feature_dir = feature_dir / cam_name
                if (sub_feature_dir / "track.npy").exists():
                    print(f"Skipping {cam_name} as it already exists.")
                    continue

                sub_feature_dir.mkdir(parents=True, exist_ok=True)
                sub_image_paths = [
                    image_path
                    for image_path in image_paths
                    if cam_name in str(image_path)
                ]

                future = executor.submit(
                    self.tracker.track, sub_image_paths, sub_feature_dir, i_proc
                )
                futures.append(future)
                print(f"Chunk {i_proc + 1}/4 submitted.")

            result = [f.result() for f in futures]

        # for i_proc, cam_name in enumerate(["1_fixed", "2_dynA", "3_dynB", "4_dynC"]):
        #     sub_feature_dir = feature_dir / cam_name
        #     sub_feature_dir.mkdir(parents=True, exist_ok=True)
        #     sub_image_paths = [
        #         image_path for image_path in image_paths if cam_name in str(image_path)
        #     ]
        #     print(f"Processing chunk {i_proc + 1}/4: {cam_name}")
        #     self.tracker.track(sub_image_paths, sub_feature_dir, i_proc)

        torch.cuda.empty_cache()
        gc.collect()

        lap_tracking = time()
        print(f"Tracking completed in {(lap_tracking - start) // 60:.2f} minutes.")

        # Merge all trajectories
        merged_track_path = feature_dir / "track_merged.npy"
        if merged_track_path.exists():
            dict_trajs = np.load(merged_track_path, allow_pickle=True).item()
            print("Using pre-merged trajectories.")
        else:
            print("Merging trajectories from all cameras.")
            dict_trajs = defaultdict(dict)
            for i_proc, cam_name in enumerate(
                ["1_fixed", "2_dynA", "3_dynB", "4_dynC"]
            ):
                trajs = (
                    np.load(feature_dir / cam_name / "track.npy", allow_pickle=True)
                    .item()
                    .trajs
                )
                dict_trajs.update(trajs)
        trajectories = TrajectorySet(dict_trajs)

        with h5py.File(feature_dir / "keypoints.h5", mode="w") as f_keypoints:
            # Get keypoints for each image from trajectories
            trajectories.build_invert_indexes()
            keypoints_per_image = {}
            kpts_indices = {}
            for frame_id, trajs_dict in trajectories.invert_maps.items():
                key = "-".join(image_paths[frame_id].parts[-3:])
                kp_list = []
                desc_list = []
                traj_list = []
                idx_in_traj_list = []
                kpts_indices_map = {}
                # NOTE: Make sure invert_maps[frame_id] are sorted by traj_id
                for traj_id, idx_in_traj in sorted(trajs_dict[frame_id].items()):
                    traj = trajectories.trajs[traj_id]
                    kp = traj.xys[idx_in_traj]  # (x, y) location at this frame
                    desc = traj.descs[idx_in_traj]
                    kp_list.append(kp)
                    desc_list.append(desc)
                    traj_list.append(traj_id)
                    idx_in_traj_list.append(idx_in_traj)  # Store index in trajectory
                    kpts_indices_map[traj_id] = len(kp_list) - 1
                keypoints_per_image[int(frame_id)] = (
                    np.array(kp_list),
                    np.array(desc_list),
                    np.array(traj_list, dtype=int),
                    np.array(idx_in_traj_list, dtype=int),
                )
                kpts_indices[int(frame_id)] = kpts_indices_map
                f_keypoints[key] = np.array(kp_list)

        index_pairs = self.retriever.get_index_pairs(
            image_paths, "exhaustive_keyframe", self.cfg.tracker.window_len
        )
        # Match trajectories for each pair: dict[frame_i][frame_j] -> (traj_id_i, traj_id_j)
        matched_traj_ids = self.matcher.match_keypoints_traj(
            image_paths, keypoints_per_image, index_pairs
        )

        torch.cuda.empty_cache()
        gc.collect()

        # Propagate matches to trajectories
        for frame_i in tqdm(matched_traj_ids):
            for frame_j in matched_traj_ids[frame_i]:
                traj_pairs = matched_traj_ids[frame_i][
                    frame_j
                ]  # List of (traj_id_i, traj_id_j)
                for traj_id_i, traj_id_j in zip(traj_pairs[0], traj_pairs[1]):
                    traj_j = trajectories.trajs[traj_id_j]
                    # Append data from traj_j to traj_i
                    trajectories.trajs[traj_id_i].times.extend(traj_j.times)
                    trajectories.trajs[traj_id_i].xys.extend(traj_j.xys)
                    # trajectories.trajs[traj_id_i].descs.extend(traj_j.descs)

        trajectories.build_invert_indexes()
        mask_dir = Path(str(image_paths[0].parent).replace("images", "masks"))
        mask_imgs = [
            cv2.imread(mask_dir / pth.name, cv2.IMREAD_GRAYSCALE) for pth in image_paths
        ]
        trajectories.build_match_indexes(
            feature_dir, mask_imgs, kpts_indices, i_proc=0
        )

        end = time()

        self.logger.log({"matching_time": (end - start) // 60})
        self.logger.summary["tracking_time"] = (lap_tracking - start) // 60
        self.logger.summary["sparse_time"] = (end - lap_tracking) // 60
