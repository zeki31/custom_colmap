import gc
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Literal

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
        save_dir: Path,
        retriever: Retriever,
    ):
        super().__init__(cfg, logger, device, save_dir, retriever)

        self.tracker = Tracker(cfg.tracker, logger, save_dir)
        self.detector = KeypointDetector(
            cfg.keypoint_detector, logger, device, save_dir
        )
        self.matcher = KeypointMatcher(cfg.keypoint_matcher, logger, save_dir)

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
            print("2. Loading pre-merged trajectories...")
            trajectories = np.load(track_path, allow_pickle=True).item()
            trajectories.build_invert_indexes()
        else:
            print("1. Merging trajectories from all cameras...")
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
            np.save(track_path, trajectories, allow_pickle=True)
            trajectories.build_invert_indexes()

        print("Register keypoints in all cameras...")
        print("1. Register keypoints in dynamic cameras...")
        kpts_per_img = self.detector.register_keypoints(
            image_paths,
            feature_dir,
            trajectories,
            self.tracker.cfg.query,
            # viz=True,
        )
        # print("2. Register keypoints in a fixed camera...")
        # self.detector.detect_keypoints(
        #     image_paths[: len(image_paths) // 4],
        #     feature_dir=feature_dir,
        #     mode="r+",
        # )

        torch.cuda.empty_cache()
        gc.collect()

        print("Match keypoints in all cameras...")
        print("1. Matching keypoints in each dynamic camera...")
        index_pairs = self.retriever.get_index_pairs(
            image_paths, "frame", self.cfg.tracker.window_len
        )
        self.matcher.traj2match(
            image_paths,
            feature_dir,
            index_pairs,
            kpts_per_img,
            # viz=True,
        )
        gc.collect()
        print("2. Matching keypoints among dynamic cameras...")
        index_pairs = self.retriever.get_index_pairs(
            image_paths, "exhaustive_keyframe", self.cfg.tracker.window_len
        )
        self.matcher.match_keypoints(image_paths, feature_dir, index_pairs)

        # print("2. Matching keypoints in the fixed camera...")
        # index_pairs = self.retriever.get_index_pairs(image_paths, "fixed")
        # self.matcher.match_keypoints_fixed(
        #     index_pairs,
        #     image_paths,
        #     feature_dir,
        #     # viz=True,
        # )

        torch.cuda.empty_cache()
        gc.collect()

        end = time()

        self.logger.log({"Matching time (min)": (end - start) // 60})
        self.logger.summary["Tracking time (min)"] = (lap_tracking - start) // 60
