import gc
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Literal

import torch
import wandb

from src.matching.matcher import Matcher
from src.matching.retriever import Retriever
from src.matching.sparse.keypoint_detector import KeypointDetector, KeypointDetectorCfg
from src.matching.sparse.keypoint_matcher import KeypointMatcher, KeypointMatcherCfg


@dataclass
class MatcherSparseCfg:
    name: Literal["sparse"]
    pair_generator: str
    keypoint_detector: KeypointDetectorCfg
    keypoint_matcher: KeypointMatcherCfg


class MatcherSparse(Matcher[MatcherSparseCfg]):
    def __init__(
        self,
        cfg: MatcherSparseCfg,
        logger: wandb.sdk.wandb_run.Run,
        device: torch.device,
        retriever: Retriever,
    ):
        super().__init__(cfg, logger, device, retriever)
        self.detector = KeypointDetector(cfg.keypoint_detector, logger, device)
        self.matcher = KeypointMatcher(cfg.keypoint_matcher, logger)

    def match(
        self,
        image_paths: list[Path],
        feature_dir: Path,
    ) -> None:
        start = time()
        self.detector.detect_keypoints(image_paths, feature_dir)
        index_pairs = self.retriever.get_index_pairs(
            image_paths, self.cfg.pair_generator
        )
        self.matcher.match_keypoints(image_paths, feature_dir, index_pairs)
        lap = time()

        torch.cuda.empty_cache()
        gc.collect()

        print(f"Matching completed in {(lap - start) // 60} minutes.")
        self.logger.log({"matching_time": (lap - start) // 60})

        index_pairs_fixed = self.retriever.get_index_pairs(image_paths, "fixed")
        self.matcher.match_keypoints_fixed(index_pairs_fixed, image_paths, feature_dir)

        torch.cuda.empty_cache()
        gc.collect()

        print(f"Matching with fixed pairs completed in {(time() - lap) // 60} minutes.")
        self.logger.log({"matching_fixed_time": (time() - lap) // 60})
