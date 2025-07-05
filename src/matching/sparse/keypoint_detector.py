from dataclasses import dataclass
from pathlib import Path

import h5py
import kornia as K
import torch
import wandb
from tqdm import tqdm

from src.submodules.LightGlue.lightglue import ALIKED


@dataclass
class KeypointDetectorCfg:
    num_features: int = 4096
    resize_to: int = 1024


class KeypointDetector:
    def __init__(
        self,
        cfg: KeypointDetectorCfg,
        logger: wandb.sdk.wandb_run.Run,
        device: torch.device,
    ):
        self.cfg = cfg
        self.logger = logger
        self.device = device

    def _load_torch_image(self, file_name: Path | str, device=torch.device("cpu")):
        """Loads an image and adds batch dimension"""
        img = K.io.load_image(file_name, K.io.ImageLoadType.RGB32, device=device)[
            None, ...
        ]
        return img

    def detect_keypoints(
        self,
        paths: list[Path],
        feature_dir: Path,
    ) -> None:
        """Detects the keypoints in a list of images with ALIKED

        Stores them in feature_dir/keypoints.h5 and feature_dir/descriptors.h5
        to be used later with LightGlue
        """
        if (feature_dir / "descriptors.h5").exists():
            return

        dtype = torch.float32  # ALIKED has issues with float16

        extractor = (
            ALIKED(
                max_num_keypoints=self.cfg.num_features,
                detection_threshold=0.01,
                resize=self.cfg.resize_to,
            )
            .eval()
            .to(self.device, dtype)
        )

        feature_dir.mkdir(parents=True, exist_ok=True)

        with h5py.File(
            feature_dir / "keypoints.h5", mode="w"
        ) as f_keypoints, h5py.File(
            feature_dir / "descriptors.h5", mode="w"
        ) as f_descriptors:
            for path in tqdm(paths, desc="Computing keypoints"):
                key = "-".join(path.parts[-3:])

                with torch.inference_mode():
                    image = self._load_torch_image(path, device=self.device).to(dtype)
                    features = extractor.extract(image)

                    f_keypoints[key] = (
                        features["keypoints"].squeeze().detach().cpu().numpy()
                    )
                    f_descriptors[key] = (
                        features["descriptors"].squeeze().detach().cpu().numpy()
                    )
