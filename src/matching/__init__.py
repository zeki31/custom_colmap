from pathlib import Path

import torch
import wandb

from src.matching.matcher import Matcher
from src.matching.matcher_sparse import MatcherSparse, MatcherSparseCfg
from src.matching.matcher_tracking import MatcherTracking, MatcherTrackingCfg
from src.matching.retriever import Retriever

MATCHERS = {
    "sparse": MatcherSparse,
    "tracking": MatcherTracking,
}

MatcherCfg = MatcherSparseCfg | MatcherTrackingCfg


def get_matcher(
    cfg: MatcherCfg,
    logger: wandb.sdk.wandb_run.Run,
    device: torch.device,
    paths: list[Path],
    feature_dir: Path,
    save_dir: Path,
    retriever: Retriever,
) -> Matcher:
    return MATCHERS[cfg.name](
        cfg, logger, device, paths, feature_dir, save_dir, retriever
    )
