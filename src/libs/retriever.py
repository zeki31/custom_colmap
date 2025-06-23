import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class RetrieverCfg:
    img_dir_type: Literal["images", "frames", "stream"]
    ext: str
    start: int
    stride: int
    duration: int


class Retriever:
    def __init__(self, cfg: RetrieverCfg):
        self.cfg = cfg

    def get_image_paths(self, base_dir: Path) -> list[Path]:
        """Get a list of image paths from the images directory."""
        if self.cfg.img_dir_type == "images":
            images_dir = base_dir / "images"
            image_paths = sorted(list(images_dir.glob(f"*.{self.cfg.ext}")))
            if self.cfg.duration > 0:
                image_paths = image_paths[
                    self.cfg.start : self.cfg.start + self.cfg.duration
                ]

        elif self.cfg.img_dir_type == "stream":
            cam_dirs = [
                "1_fixed/images",
                "2_dynA/images",
                "3_dynB/images",
                "4_dynC/images",
            ]
            if self.cfg.duration > 0:
                image_paths = [
                    img_path
                    for folder in cam_dirs
                    for img_path in sorted(
                        (base_dir / folder).glob(f"*.{self.cfg.ext}")
                    )[self.cfg.start : self.cfg.start + self.cfg.duration]
                ]
            else:
                image_paths = [
                    img_path
                    for folder in cam_dirs
                    for img_path in sorted(
                        (base_dir / folder).glob(f"*.{self.cfg.ext}")
                    )[self.cfg.start :]
                ]

        if self.cfg.stride > 1:
            image_paths = image_paths[:: self.cfg.stride]

        print(f"Got {len(image_paths)} images")

        return image_paths

    def get_pairs_exhaustive(
        self,
        paths: list[Path],
    ) -> list[tuple[int, int]]:
        """Obtains all possible index pairs of a list"""
        return list(itertools.combinations(range(len(paths)), 2))
