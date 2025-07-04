import torch
import wandb

from src.matching.matcher import Matcher
from src.matching.matcher_sparse import MatcherSparse, MatcherSparseCfg
from src.matching.matcher_tracking import MatcherTracking, MatcherTrackingCfg

MATCHERS = {
    "sparse": MatcherSparse,
    "tracking": MatcherTracking,
}

MatcherCfg = MatcherSparseCfg | MatcherTrackingCfg


def get_matcher(
    cfg: MatcherCfg, logger: wandb.sdk.wandb_run.Run, device: torch.device
) -> Matcher:
    return MATCHERS[cfg.name](cfg, logger, device)
