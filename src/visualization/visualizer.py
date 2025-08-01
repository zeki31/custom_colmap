from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

import wandb

T = TypeVar("T")


class Visualizer(ABC, Generic[T]):
    def __init__(
        self,
        cfg: T,
        logger: wandb.sdk.wandb_run.Run,
        save_dir: Path,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.logger = logger
        self.save_dir = save_dir

    @abstractmethod
    def viz(self, base_dir: Path) -> None:
        pass
