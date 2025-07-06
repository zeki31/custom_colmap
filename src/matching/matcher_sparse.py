from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Literal

import torch
import wandb

from src.matching.matcher import Matcher
from src.matching.sparse.keypoint_detector import KeypointDetector, KeypointDetectorCfg
from src.matching.sparse.keypoint_matcher import KeypointMatcher, KeypointMatcherCfg


@dataclass
class MatcherSparseCfg:
    name: Literal["sparse"]
    keypoint_detector: KeypointDetectorCfg
    keypoint_matcher: KeypointMatcherCfg


class MatcherSparse(Matcher):
    def __init__(
        self,
        cfg: MatcherSparseCfg,
        logger: wandb.sdk.wandb_run.Run,
        device: torch.device,
    ):
        self.cfg = cfg
        self.logger = logger
        self.device = device

        self.detector = KeypointDetector(cfg.keypoint_detector, logger, device)
        self.matcher = KeypointMatcher(cfg.keypoint_matcher, logger)

    def match(
        self,
        image_paths: list[Path],
        feature_dir: Path,
    ) -> None:
        start = time()
        self.detector.detect_keypoints(image_paths, feature_dir)
        self.matcher.match_keypoints(image_paths, feature_dir)
        end = time()

        print(f"Matching completed in {(end - start) // 60} minutes.")
        self.logger.log({"matching_time": (end - start) // 60})
