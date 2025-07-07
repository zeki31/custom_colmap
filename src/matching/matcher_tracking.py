import multiprocessing as mp

mp.set_start_method("spawn", force=True)

from concurrent.futures import ProcessPoolExecutor
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


@dataclass
class MatcherTrackingCfg:
    name: Literal["tracking"]
    tracker: TrackerCfg
    keypoint_detector: KeypointDetectorCfg
    keypoint_matcher: KeypointMatcherCfg


class MatcherTracking(Matcher):
    def __init__(
        self,
        cfg: MatcherTrackingCfg,
        logger: wandb.sdk.wandb_run.Run,
        device: torch.device,
        retriever: Retriever,
    ):
        super().__init__(cfg, logger, device, retriever)

        self.tracker = Tracker(cfg.tracker, logger, device)
        self.detector = KeypointDetector(cfg.keypoint_detector, logger, device)
        self.matcher = KeypointMatcher(cfg.keypoint_matcher, logger)

    def match(self, image_paths: list[Path], feature_dir: Path) -> None:
        start = time()
        futures = []
        with ProcessPoolExecutor(max_workers=2) as executor:
            for i_proc, cam_name in enumerate(
                ["1_fixed", "2_dynA", "3_dynB", "4_dynC"]
            ):
                sub_feature_dir = feature_dir / cam_name
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
        lap_tracking = time()

        # Merge all trajectories
        dict_trajs = {}
        for i_proc, cam_name in enumerate(["1_fixed", "2_dynA", "3_dynB", "4_dynC"]):
            trajs = np.load(
                feature_dir / cam_name / "track.npy", allow_pickle=True
            ).trajs
            dict_trajs.update(trajs)
        trajectories = TrajectorySet(dict_trajs)
        trajectories.build_invert_indexes()

        index_pairs = self.retriever.get_index_pairs(
            image_paths, "exhaustive_keyframe", self.cfg.tracker.window_len
        )
        self.matcher.match_keypoints(image_paths, feature_dir, index_pairs)
        end = time()

        self.logger.log({"matching_time": (end - start) // 60})
        self.logger.summary["tracking_time"] = (lap_tracking - start) // 60
        self.logger.summary["sparse_time"] = (end - lap_tracking) // 60
