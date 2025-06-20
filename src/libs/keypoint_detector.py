from dataclasses import dataclass
from pathlib import Path

import h5py
import kornia as K
import torch
from tqdm import tqdm

from .submodules.LightGlue.lightglue import ALIKED


@dataclass
class KeypointDetectorCfg:
    num_features: int = 4096
    resize_to: int = 1024


def load_torch_image(file_name: Path | str, device=torch.device("cpu")):
    """Loads an image and adds batch dimension"""
    img = K.io.load_image(file_name, K.io.ImageLoadType.RGB32, device=device)[None, ...]
    return img


def detect_keypoints(
    paths: list[Path],
    feature_dir: Path,
    cfg: KeypointDetectorCfg,
    device: torch.device = torch.device("cpu"),
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
            max_num_keypoints=cfg.num_features,
            detection_threshold=0.01,
            resize=cfg.resize_to,
        )
        .eval()
        .to(device, dtype)
    )

    feature_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(feature_dir / "keypoints.h5", mode="w") as f_keypoints, h5py.File(
        feature_dir / "descriptors.h5", mode="w"
    ) as f_descriptors:
        for path in tqdm(paths, desc="Computing keypoints"):
            key = path.parts[-3] + "-images-" + path.name

            with torch.inference_mode():
                image = load_torch_image(path, device=device).to(dtype)
                features = extractor.extract(image)

                f_keypoints[key] = (
                    features["keypoints"].squeeze().detach().cpu().numpy()
                )
                f_descriptors[key] = (
                    features["descriptors"].squeeze().detach().cpu().numpy()
                )
