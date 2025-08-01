from pathlib import Path

import wandb

from src.visualization.visualizer import Visualizer
from src.visualization.visualizer_rerun import VisualizerRerun, VisualizerRerunCfg
from src.visualization.visualizer_viser import VisualizerViser, VisualizerViserCfg

VISUALIZERS = {
    "rerun": VisualizerRerun,
    "viser": VisualizerViser,
}

VisualizerCfg = VisualizerRerunCfg | VisualizerViserCfg


def get_visualizer(
    cfg: VisualizerCfg,
    logger: wandb.sdk.wandb_run.Run,
    save_dir: Path,
) -> Visualizer:
    return VISUALIZERS[cfg.name](cfg, logger, save_dir)
