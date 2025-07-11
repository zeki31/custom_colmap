import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import kornia as K
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from jaxtyping import Float, Int
from numpy.typing import NDArray
from tqdm import tqdm

from src.matching.tracking.trajectory import TrajectorySet
from src.submodules.LightGlue.lightglue import ALIKED, viz2d
from src.submodules.LightGlue.lightglue.utils import load_image


@dataclass
class KeypointDetectorCfg:
    num_features: int = 4096


class KeypointDetector:
    def __init__(
        self,
        cfg: KeypointDetectorCfg,
        logger: wandb.sdk.wandb_run.Run,
        device: torch.device,
        save_dir: Path,
    ):
        self.cfg = cfg
        self.logger = logger
        self.device = device
        self.save_dir = save_dir

        self.dtype = torch.float32  # ALIKED has issues with float16
        self.extractor = (
            ALIKED(
                max_num_keypoints=self.cfg.num_features,
                detection_threshold=0.01,
                # resize=self.cfg.resize_to,
            )
            .eval()
            .to(self.device, self.dtype)
        )

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
        mode: Literal["w", "a"] = "w",
    ) -> None:
        """Detects the keypoints in a list of images with ALIKED

        Stores them in feature_dir/keypoints.h5 and feature_dir/descriptors.h5
        to be used later with LightGlue
        """
        if (feature_dir / "descriptors.h5").exists():
            return

        with h5py.File(
            feature_dir / "keypoints.h5", mode=mode
        ) as f_keypoints, h5py.File(
            feature_dir / "descriptors.h5", mode=mode
        ) as f_descriptors:
            for path in tqdm(paths, desc="Computing keypoints"):
                key = "-".join(path.parts[-3:])

                with torch.inference_mode():
                    image = self._load_torch_image(path, device=self.device).to(
                        self.dtype
                    )
                    features = self.extractor.extract(image)

                    f_keypoints[key] = (
                        features["keypoints"].squeeze().detach().cpu().numpy()
                    )
                    f_descriptors[key] = (
                        features["descriptors"].squeeze().detach().cpu().numpy()
                    )

    def register_keypoints(
        self,
        paths: list[Path],
        feature_dir: Path,
        trajectories: TrajectorySet,
        query: Literal["grid", "aliked"],
        viz: bool = False,
    ) -> dict[int, tuple[Float[NDArray, "... 2"], Int[NDArray, "..."]]]:
        """Detects the keypoints in a list of images with ALIKED

        Stores them in feature_dir/keypoints.h5 and feature_dir/descriptors.h5
        to be used later with LightGlue
        """
        if viz:
            viz_dir = self.save_dir / "keypoints_viz"
            viz_dir.mkdir(parents=True, exist_ok=True)

        with h5py.File(
            feature_dir / "keypoints.h5", mode="r+"
        ) as f_keypoints, h5py.File(
            feature_dir / "descriptors.h5", mode="r+"
        ) as f_descriptors:
            kpts_per_img = {}
            for frame_id, trajs_dict in tqdm(
                sorted(trajectories.invert_maps.items()), desc="Registering keypoints"
            ):
                key = "-".join(paths[frame_id].parts[-3:])
                kpts = []
                descs = []
                traj_ids = []
                for traj_id, idx_in_traj in sorted(trajs_dict.items()):
                    traj = trajectories.trajs[traj_id]
                    traj_ids.append(traj_id)
                    kpts.append(traj.xys[idx_in_traj])
                    if query == "aliked":
                        descs.append(traj.descs[idx_in_traj])
                kpts_np = np.stack(kpts, dtype=np.float32) + 0.5
                if query == "aliked":
                    descs_np = np.stack(descs, dtype=np.float32)

                if viz:
                    image0 = load_image(paths[frame_id])
                    viz2d.plot_images([image0])
                    viz2d.plot_keypoints([kpts_np], ps=10)
                    viz2d.add_text(
                        0, paths[frame_id].parts[-3] + "_" + paths[frame_id].name
                    )
                    viz2d.save_plot(viz_dir / f"{key}.png")
                    plt.close()

                f_keypoints[key] = kpts_np
                if query == "aliked":
                    f_descriptors[key] = descs_np
                kpts_per_img[int(frame_id)] = (
                    kpts_np,
                    np.array(traj_ids, dtype=int),
                )

        if viz:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    "12",
                    "-i",
                    str(viz_dir / "%*.png"),
                    "-c:v",
                    "libx264",
                    str(viz_dir / "kpts.mp4"),
                ]
            )
            self.logger.log(
                {
                    "Keypoint visualization": wandb.Video(
                        str(viz_dir / "kpts.mp4"), format="mp4"
                    )
                }
            )

        return kpts_per_img
