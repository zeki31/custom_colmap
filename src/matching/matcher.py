from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

import torch
import wandb

T = TypeVar("T")


class Matcher(ABC, Generic[T]):
    def __init__(
        self, cfg: T, logger: wandb.sdk.wandb_run.Run, device: torch.device
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.logger = logger
        self.device = device

    @abstractmethod
    def match(self, image_paths: list[Path], feature_dir: Path) -> None:
        pass
