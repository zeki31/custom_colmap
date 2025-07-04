from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Literal

import torch
import wandb

from src.matching.matcher import Matcher
from src.matching.sparse.keypoint_detector import KeypointDetector, KeypointDetectorCfg
from src.matching.sparse.keypoint_matcher import KeypointMatcher, KeypointMatcherCfg
from src.matching.tracking.tracker import Tracker, TrackerCfg


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
    ):
        self.cfg = cfg
        self.logger = logger
        self.device = device

        self.tracker = Tracker(cfg.tracker, logger, device)
        self.detector = KeypointDetector(cfg.keypoint_detector, logger, device)
        self.matcher = KeypointMatcher(cfg.keypoint_matcher, logger, device)

    def match(self, image_paths: list[Path], feature_dir: Path) -> None:
        start = time()
        self.tracker.track(image_paths, feature_dir)
        lap_tracking = time()

        self.detector.detect_keypoints(image_paths, feature_dir)
        self.matcher.keypoint_distances(image_paths, feature_dir)
        end = time()

        self.logger.log({"matching_time": (end - start) // 60})
        self.logger.summary({"tracking_time": (lap_tracking - start) // 60})
        self.logger.summary({"sparse_time": (end - lap_tracking) // 60})
