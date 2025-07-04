from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import cv2
import wandb
from tqdm import tqdm


@dataclass
class RetrieverCfg:
    img_dir_type: Literal["images", "frames", "stream"]
    ext: str
    start: int
    stride: int
    duration: int

    comp_ratio: Optional[int]


class Retriever:
    def __init__(self, cfg: RetrieverCfg, logger: wandb.sdk.wandb_run.Run):
        self.cfg = cfg
        self.logger = logger

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
        self.logger.summary["n_images"] = len(image_paths)

        if self.cfg.comp_ratio is None:
            return image_paths

        image_paths_resized = [
            Path(str(img_path).replace("images", "images_resized"))
            for img_path in image_paths
        ]
        if not image_paths == [] and image_paths_resized[0].exists():
            return image_paths_resized

        image_paths_resized[0].parent.mkdir(parents=True, exist_ok=True)
        image_paths_resized = []
        for img_path in tqdm(image_paths, desc="Resizing images"):
            img = cv2.imread(str(img_path))
            img_resized = cv2.resize(
                img,
                (
                    img.shape[1] // self.cfg.comp_ratio,
                    img.shape[0] // self.cfg.comp_ratio,
                ),
            )

            img_path_resized = Path(str(img_path).replace("images", "images_resized"))
            if not img_path_resized.parent.exists():
                img_path_resized.parent.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(img_path_resized), img_resized)
            image_paths_resized.append(img_path_resized)

        return image_paths_resized
