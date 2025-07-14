import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import cv2
import h5py
import kornia as K
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from jaxtyping import Float, Int
from numpy.typing import NDArray
from tqdm import tqdm
from tqdm.contrib import tzip

from src.matching.tracking.trajectory import Trajectory, TrajectorySet
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
        mode: Literal["w", "r+"] = "w",
    ) -> None:
        """Detects the keypoints in a list of images with ALIKED

        Stores them in feature_dir/keypoints.h5 and feature_dir/descriptors.h5
        to be used later with LightGlue
        """
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

    def detect_keypoints_fixed(
        self,
        paths: list[Path],
        feature_dir: Path,
    ) -> None:
        mask_imgs = [
            cv2.imread(
                Path(str(path.parent).replace("images", "masks")) / path.name,
                cv2.IMREAD_GRAYSCALE,
            )
            for path in paths
        ]
        merged_mask = mask_imgs[0]
        for mask in mask_imgs:
            merged_mask = np.logical_or(merged_mask, mask).astype(np.uint8) * 255

        with h5py.File(
            feature_dir / "keypoints.h5", mode="r+"
        ) as f_keypoints, h5py.File(
            feature_dir / "descriptors.h5", mode="r+"
        ) as f_descriptors:
            key = "-".join(paths[0].parts[-3:])
            with torch.inference_mode():
                image = self._load_torch_image(paths[0], device=self.device).to(
                    self.dtype
                )
                features = self.extractor.extract(image)

                kpts = features["keypoints"].squeeze().detach().cpu().numpy()
                descs = features["descriptors"].squeeze().detach().cpu().numpy()
                masked_idx = np.where(
                    merged_mask[kpts[:, 1].astype(int), kpts[:, 0].astype(int)] == 0
                )[0]

                f_keypoints[key] = kpts[masked_idx]
                f_descriptors[key] = descs[masked_idx]

    def register_keypoints(
        self,
        paths: list[Path],
        feature_dir: Path,
        trajectories: TrajectorySet,
        query: Literal["grid", "aliked"],
        viz: bool = False,
    ) -> dict[
        int,
        tuple[
            Float[NDArray, "... 2"],
            Optional[Float[NDArray, "... D"]],
            Int[NDArray, "..."],
        ],
    ]:
        """Detects the keypoints in a list of images with ALIKED

        Stores them in feature_dir/keypoints.h5 and feature_dir/descriptors.h5
        to be used later with LightGlue
        """
        if viz:
            viz_dir = self.save_dir / "keypoints_viz"
            viz_dir.mkdir(parents=True, exist_ok=True)

        with h5py.File(
            feature_dir / "keypoints.h5", mode="w"
        ) as f_keypoints, h5py.File(
            feature_dir / "descriptors.h5", mode="w"
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
                    descs_np if query == "aliked" else None,
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
            for img in viz_dir.glob("*.png"):
                img.unlink()

        return kpts_per_img

    def track_fixed(
        self,
        paths: list[Path],
        feature_dir: Path,
        viz: bool = False,
    ) -> None:
        """Detects the keypoints in a list of images with ALIKED"""
        save_dir = feature_dir / "1_fixed"
        save_dir.mkdir(parents=True, exist_ok=True)

        if (save_dir / "full_trajs.npy").exists():
            print("Already tracked keypoints in the fixed camera, skipping.")
            return

        mask_imgs = [
            cv2.imread(
                Path(str(path.parent).replace("images", "masks")) / path.name,
                cv2.IMREAD_GRAYSCALE,
            )
            for path in paths
        ]
        merged_mask = mask_imgs[0]
        for mask in mask_imgs:
            merged_mask = np.logical_or(merged_mask, mask).astype(np.uint8) * 255

        if viz:
            cv2.imwrite(save_dir / "mask.png", merged_mask)
            self.logger.log(
                {
                    "Merged Mask": wandb.Image(
                        save_dir / "mask.png",
                    )
                }
            )

        with torch.inference_mode():
            image = self._load_torch_image(paths[0], device=self.device).to(self.dtype)
            features = self.extractor.extract(image)

            kpts = features["keypoints"].squeeze().detach().cpu().numpy()
            descs = features["descriptors"].squeeze().detach().cpu().numpy()
            masked_idx = np.where(
                merged_mask[kpts[:, 1].astype(int), kpts[:, 0].astype(int)] == 0
            )[0]
            kpts_masked = kpts[masked_idx]
            descs_masked = descs[masked_idx]

            if viz:
                viz2d.plot_images([load_image(paths[0])])
                viz2d.plot_keypoints([kpts_masked], ps=10)
                viz2d.add_text(0, paths[0].parts[-3] + "_" + paths[0].name)
                viz2d.save_plot(save_dir / "kpts.png")
                plt.close()
                self.logger.log(
                    {
                        "Keypoints Fixed": wandb.Image(
                            save_dir / "kpts.png",
                        )
                    }
                )

        full_trajs = []
        for kpts, descs in tzip(kpts_masked, descs_masked):
            traj = Trajectory(0, kpts, descs)
            full_trajs.append(traj)
        np.save(save_dir / "full_trajs.npy", full_trajs)
