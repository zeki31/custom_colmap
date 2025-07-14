import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import wandb
from tqdm.contrib import tenumerate


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
                    )[
                        self.cfg.start : self.cfg.start
                        + self.cfg.duration : self.cfg.stride
                    ]
                ]
            else:
                image_paths = [
                    img_path
                    for folder in cam_dirs
                    for img_path in sorted(
                        (base_dir / folder).glob(f"*.{self.cfg.ext}")
                    )[self.cfg.start :: self.cfg.stride]
                ]

        print(f"Got {len(image_paths)} images")
        self.logger.summary["n_images"] = len(image_paths)

        if self.cfg.comp_ratio == 1:
            return image_paths

        image_paths_resized = self._resize_imgs(image_paths, "images")
        # mask_paths = [
        #     Path(str(path.parent).replace("images", "masks")) / path.name
        #     for path in image_paths
        # ]
        # # _ = self._resize_imgs(mask_paths, "masks")

        return image_paths_resized

    def _resize_imgs(
        self, image_paths: list[Path], label: Literal["images", "masks"]
    ) -> list[Path]:
        """Resize images to the compression ratio."""
        image_paths_resized = []
        n_frames = len(image_paths) // 4
        for i, img_path in tenumerate(image_paths, desc="Resizing images"):
            if i % n_frames == 0:
                resized_dir = img_path.parents[1] / f"{label}_resized"
                resized_dir.mkdir(parents=True, exist_ok=True)

            # if label == "masks":
            #     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # else:
            #     img = cv2.imread(img_path)
            # img_resized = cv2.resize(
            #     img.copy(),
            #     (
            #         img.shape[1] // self.cfg.comp_ratio,
            #         img.shape[0] // self.cfg.comp_ratio,
            #     ),
            # )

            img_path_resized = resized_dir / img_path.name
            # cv2.imwrite(img_path_resized, img_resized)
            image_paths_resized.append(img_path_resized)

        return image_paths_resized

    def get_index_pairs(
        self,
        paths: list[Path],
        pair_generator: str,
        window_len: Optional[int] = None,
    ) -> list[tuple[int, int]]:
        if pair_generator == "exhaustive":
            # Obtains all possible index pairs of a list
            pairs = list(itertools.combinations(range(len(paths)), 2))

        if pair_generator == "exhaustive_dynamic":
            # Obtains all possible index pairs only for dynamic cameras
            dyn_cam_indices = range(len(paths) // 4, len(paths))
            pairs = list(itertools.combinations(dyn_cam_indices, 2))
            print(f"Pairs among dynamic cameras: {len(pairs)}")

        elif pair_generator == "frame-view":
            # Obtains only adjacent pairs
            # (different timestamp, same camera)
            pairs = []
            for i in range(len(paths) - 1):
                pairs.append((i, i + 1))
            # Collect the every self.cfg.duration-th pair
            # (same timestamp, different cameras)
            n_frames = len(paths) // 4
            for t in range(n_frames):
                pairs.extend(
                    list(itertools.combinations(range(t, len(paths), n_frames), 2))
                )
            pairs = sorted(set(pairs))  # Remove duplicates

        elif pair_generator == "view":
            # Obtains only adjacent cameras
            # (same timestamp, different cameras)
            pairs = []
            n_frames = len(paths) // 4
            for t in range(n_frames):
                pairs.extend(
                    list(itertools.combinations(range(t, len(paths), n_frames), 2))
                )
            pairs = sorted(set(pairs))  # Remove duplicates

        elif pair_generator == "exhaustive_keyframe_excluding_same_view":
            n_frames = len(paths) // 4

            keyframe_indices = range(n_frames, len(paths), window_len)
            pairs = list(itertools.combinations(keyframe_indices, 2))
            # Exclude pairs from the same view
            pairs = [
                pair for pair in pairs if (pair[0] // n_frames) != (pair[1] // n_frames)
            ]

            pairs = sorted(set(pairs))  # Remove duplicates
            print(f"Keyframe pairs (excluding pairs from the same view): {len(pairs)}")

        elif pair_generator == "fixed":
            # Obtains only adjacent cameras with the fixed camera
            # (same timestamp, different cameras)
            pairs = []
            n_frames = len(paths) // 4
            for t in range(n_frames):
                pairs.extend([(t, t + i * n_frames) for i in range(1, 4)])
            pairs = sorted(set(pairs))  # Remove duplicates
            print(f"Pairs with fixed camera (same frame): {len(pairs)}")

        elif pair_generator == "frame":
            pairs = []
            n_frames = len(paths) // 4
            for i in range(n_frames, n_frames * 4, n_frames):
                pairs.extend(list(itertools.combinations(range(i, i + n_frames), 2)))
            pairs = sorted(set(pairs))  # Remove duplicates
            print(f"Frame pairs (same camera): {len(pairs)}")

        return pairs
