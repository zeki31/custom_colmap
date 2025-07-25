import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import h5py
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from jaxtyping import Float, Int, UInt
from numpy.typing import NDArray
from tqdm import tqdm

from src.submodules.LightGlue.lightglue import viz2d
from src.submodules.LightGlue.lightglue.utils import load_image

mp.set_start_method("spawn", force=True)


@dataclass
class KeypointMatcherCfg:
    min_matches: int = 15
    verbose: bool = True
    mask: bool = False
    filter_threshold: float = 0.1


class KeypointMatcher:
    def __init__(
        self,
        cfg: KeypointMatcherCfg,
        logger: wandb.sdk.wandb_run.Run,
        paths: list[Path],
        feature_dir: Path,
        save_dir: Path,
    ):
        self.cfg = cfg
        self.logger = logger
        self.paths = paths
        self.feature_dir = feature_dir
        self.save_dir = save_dir

        self.n_frames = len(self.paths) // 4
        self.matcher_params = {
            "width_confidence": -1,
            "depth_confidence": -1,
            "mp": True,
            "filter_threshold": cfg.filter_threshold,
        }

        if self.cfg.mask:
            self.mask_imgs = [
                cv2.imread(
                    Path(str(path.parent).replace("images", "masks")) / path.name,
                    cv2.IMREAD_GRAYSCALE,
                )
                for path in paths
            ]

    def _chunkify(
        self,
        pairs: list[tuple[int, int]],
        n_cpu: int,
    ) -> list[list[tuple[int, int]]]:
        """Splits an pairs into n_cpu chunks."""
        print(f"Chunking pairs into {n_cpu} chunks.")
        chunk_size = len(pairs) // n_cpu + 1
        return [pairs[i : i + chunk_size] for i in range(0, len(pairs), chunk_size)]

    def _masking(
        self,
        mask_img1: UInt[NDArray, "H W"],
        mask_img2: UInt[NDArray, "H W"],
        kpts1: Float[NDArray, "... 2"],
        kpts2: Float[NDArray, "... 2"],
        descs1: Float[NDArray, "... 128"],
        descs2: Float[NDArray, "... 128"],
    ) -> tuple[
        Float[NDArray, "... 2"],
        Float[NDArray, "... 2"],
        Float[NDArray, "... 128"],
        Float[NDArray, "... 128"],
        Int[NDArray, "..."],
        Int[NDArray, "..."],
    ]:
        """Apply masks to keypoints and descriptors"""
        keep_idx1 = np.where(
            mask_img1[kpts1[:, 1].astype(int), kpts1[:, 0].astype(int)] == 0
        )[0]
        keep_idx2 = np.where(
            mask_img2[kpts2[:, 1].astype(int), kpts2[:, 0].astype(int)] == 0
        )[0]
        kpts1_masked = kpts1[keep_idx1]
        kpts2_masked = kpts2[keep_idx2]
        descs1_masked = descs1[keep_idx1]
        descs2_masked = descs2[keep_idx2]

        return (
            kpts1_masked,
            kpts2_masked,
            descs1_masked,
            descs2_masked,
            keep_idx1,
            keep_idx2,
        )

    def multiprocess(
        self,
        func,
        pairs: list[tuple[int, int]],
        n_cpu: int,
        cond: bool = False,
        *args,
    ):
        """Run a function in parallel with multiple processes."""
        if cond:
            return

        chunks = self._chunkify(pairs, n_cpu)
        with ProcessPoolExecutor(max_workers=n_cpu) as executor:
            futures = [
                executor.submit(func, chunk, i_proc, *args)
                for i_proc, chunk in enumerate(chunks)
            ]
            return [f.result() for f in futures]

    def match_keypoints(
        self,
        index_pairs: list[tuple[int, int]],
        i_proc: int,
    ):
        device = torch.device(
            f"cuda:{i_proc % 2}" if torch.cuda.is_available() else "cpu"
        )
        _matcher = KF.LightGlueMatcher("aliked", self.matcher_params).eval().to(device)

        with h5py.File(
            self.feature_dir / "keypoints.h5", mode="r"
        ) as f_keypoints, h5py.File(
            self.feature_dir / "descriptors.h5", mode="r"
        ) as f_descriptors, h5py.File(
            (self.feature_dir / f"matches_{i_proc}.h5"), mode="w"
        ) as f_matches:
            for idx1, idx2 in tqdm(
                index_pairs, desc=f"Matching keypoints in the process {i_proc}"
            ):
                key1 = "-".join(self.paths[idx1].parts[-3:])
                key2 = "-".join(self.paths[idx2].parts[-3:])

                kpts1 = f_keypoints[key1][...]
                kpts2 = f_keypoints[key2][...]
                descs1 = f_descriptors[key1][...]
                descs2 = f_descriptors[key2][...]

                # Mask out keypoints in dynamic region before matching
                if self.cfg.mask and idx1 % self.n_frames != idx2 % self.n_frames:
                    kpts1, kpts2, descs1, descs2, keep_idx1, keep_idx2 = self._masking(
                        self.mask_imgs[idx1],
                        self.mask_imgs[idx2],
                        kpts1,
                        kpts2,
                        descs1,
                        descs2,
                    )

                keypoints1 = torch.from_numpy(kpts1).to(device)
                keypoints2 = torch.from_numpy(kpts2).to(device)
                descriptors1 = torch.from_numpy(descs1).to(device)
                descriptors2 = torch.from_numpy(descs2).to(device)

                with torch.inference_mode():
                    _, indices = _matcher(
                        descriptors1,
                        descriptors2,
                        KF.laf_from_center_scale_ori(keypoints1[None]),
                        KF.laf_from_center_scale_ori(keypoints2[None]),
                    )
                indices = indices.detach().cpu().numpy().reshape(-1, 2)
                if self.cfg.mask and idx1 % self.n_frames != idx2 % self.n_frames:
                    indices = np.stack(
                        [
                            keep_idx1[indices[:, 0]],
                            keep_idx2[indices[:, 1]],
                        ],
                        axis=1,
                    )

                if len(indices) >= self.cfg.min_matches:
                    group = f_matches.require_group(key1)
                    group.create_dataset(key2, data=indices)

    def match_trajectories(
        self,
        index_pairs: list[tuple[int, int]],
        i_proc: int,
        kpts_per_img: dict[
            int,
            tuple[
                Float[NDArray, "... 2"],
                Float[NDArray, "... D"],
                Int[NDArray, "..."],
            ],
        ],
        # viz: bool = False,
    ) -> set[tuple[int, int]]:
        """Match trajectories in the different dynamic cameras."""
        device = torch.device(
            f"cuda:{i_proc % 2}" if torch.cuda.is_available() else "cpu"
        )
        _matcher = KF.LightGlueMatcher("aliked", self.matcher_params).eval().to(device)

        # if viz:
        #     viz_dir = self.save_dir / "matched_trajs_viz"
        #     viz_dir.mkdir(parents=True, exist_ok=True)

        traj_pairs = []
        for idx1, idx2 in tqdm(
            index_pairs, desc="Matching trajectories in different dynamic cameras"
        ):
            kpts1, descs1, traj_ids1 = kpts_per_img[idx1]
            kpts2, descs2, traj_ids2 = kpts_per_img[idx2]

            # Mask out keypoints in dynamic region before matching
            if self.cfg.mask and idx1 % self.n_frames != idx2 % self.n_frames:
                kpts1, kpts2, descs1, descs2, keep_idx1, keep_idx2 = self._masking(
                    self.mask_imgs[idx1],
                    self.mask_imgs[idx2],
                    kpts1,
                    kpts2,
                    descs1,
                    descs2,
                )

            keypoints1 = torch.from_numpy(kpts1).to(device)
            keypoints2 = torch.from_numpy(kpts2).to(device)
            descriptors1 = torch.from_numpy(descs1).to(device)
            descriptors2 = torch.from_numpy(descs2).to(device)

            with torch.inference_mode():
                _, indices = _matcher(
                    descriptors1,
                    descriptors2,
                    KF.laf_from_center_scale_ori(keypoints1[None]),
                    KF.laf_from_center_scale_ori(keypoints2[None]),
                )
            indices = indices.detach().cpu().numpy().reshape(-1, 2)
            if self.cfg.mask and idx1 % self.n_frames != idx2 % self.n_frames:
                indices = np.stack(
                    [
                        keep_idx1[indices[:, 0]],
                        keep_idx2[indices[:, 1]],
                    ],
                    axis=1,
                )

            # if viz:
            #     image1 = load_image(self.paths[idx1])
            #     image2 = load_image(self.paths[idx2])
            #     viz2d.plot_images([image1, image2])
            #     viz2d.plot_matches(
            #         keypoints1[indices[:, 0]],
            #         keypoints2[indices[:, 1]],
            #         color="lime",
            #         lw=0.2,
            #     )
            #     key1_viz = self.paths[idx1].parts[-3] + "_" + self.paths[idx1].stem
            #     key2_viz = self.paths[idx2].parts[-3] + "_" + self.paths[idx2].stem
            #     viz2d.add_text(0, key1_viz, fs=20)
            #     viz2d.add_text(1, key2_viz, fs=20)
            #     viz2d.save_plot(viz_dir / f"{key1_viz}_{key2_viz}.png")
            #     plt.close()

            if len(indices):
                matched_traj_ids = np.stack(
                    [traj_ids1[indices[:, 0]], traj_ids2[indices[:, 1]]], axis=1
                )
                traj_pairs.extend(matched_traj_ids.tolist())

        # if viz:
        #     subprocess.run(
        #         [
        #             "ffmpeg",
        #             "-y",
        #             "-framerate",
        #             "12",
        #             "-i",
        #             str(viz_dir / "%*.png"),
        #             "-c:v",
        #             "libx264",
        #             str(viz_dir / "matches.mp4"),
        #         ]
        #     )
        #     wandb.log(
        #         {
        #             "Matched Trajectories": wandb.Video(
        #                 str(viz_dir / "matches.mp4"), format="mp4"
        #             )
        #         }
        #     )
        #     for img in viz_dir.glob("*.png"):
        #         img.unlink()

        return set(map(tuple, traj_pairs))

    def traj2match(
        self,
        index_pairs: list[tuple[int, int]],
        i_proc: int,
        kpts_per_img: dict[
            int,
            tuple[
                Float[NDArray, "... 2"],
                Optional[Float[NDArray, "... D"]],
                Int[NDArray, "..."],
            ],
        ],
        viz: bool = False,
    ) -> None:
        """Match keypoints in the dynamic cameras exhaustively."""
        if viz:
            viz_dir = self.save_dir / "matches_viz"
            viz_dir.mkdir(parents=True, exist_ok=True)
            images = [load_image(path) for path in self.paths]

        with h5py.File(
            self.feature_dir / f"matches_{i_proc}.h5", mode="w"
        ) as f_matches:
            for idx1, idx2 in tqdm(
                index_pairs, desc="Converting trajectories to matches"
            ):
                key1 = "-".join(self.paths[idx1].parts[-3:])
                key2 = "-".join(self.paths[idx2].parts[-3:])

                kpts1, _, traj_ids1 = kpts_per_img[idx1]
                kpts2, _, traj_ids2 = kpts_per_img[idx2]
                _, idx1_of_common, idx2_of_common = np.intersect1d(
                    traj_ids1, traj_ids2, assume_unique=True, return_indices=True
                )
                indices = np.stack([idx1_of_common, idx2_of_common], axis=1)

                # Mask out keypoints that are in a dynamic region
                if self.cfg.mask and idx1 % self.n_frames != idx2 % self.n_frames:
                    mask1 = self.mask_imgs[idx1]
                    mask2 = self.mask_imgs[idx2]
                    m_kpts1 = kpts1[indices[:, 0]]
                    m_kpts2 = kpts2[indices[:, 1]]
                    mask_val1 = mask1[
                        m_kpts1[:, 1].astype(int), m_kpts1[:, 0].astype(int)
                    ]
                    mask_val2 = mask2[
                        m_kpts2[:, 1].astype(int), m_kpts2[:, 0].astype(int)
                    ]
                    indices = indices[(mask_val1 == 0) & (mask_val2 == 0)].copy()

                if viz:
                    image1 = images[idx1]
                    image2 = images[idx2]
                    viz2d.plot_images([image1, image2])
                    viz2d.plot_matches(
                        kpts1[indices[:, 0]], kpts2[indices[:, 1]], color="lime", lw=0.2
                    )
                    key1_viz = self.paths[idx1].parts[-3] + "_" + self.paths[idx1].stem
                    key2_viz = self.paths[idx2].parts[-3] + "_" + self.paths[idx2].stem
                    viz2d.add_text(0, key1_viz, fs=20)
                    viz2d.add_text(1, key2_viz, fs=20)
                    viz2d.save_plot(viz_dir / f"{key1_viz}_{key2_viz}.png")
                    plt.close()

                if len(indices) >= self.cfg.min_matches:
                    group = f_matches.require_group(key1)
                    group.create_dataset(key2, data=indices)

        # if viz:
        #     subprocess.run(
        #         [
        #             "ffmpeg",
        #             "-y",
        #             "-framerate",
        #             "12",
        #             "-i",
        #             str(viz_dir / "%*.png"),
        #             "-c:v",
        #             "libx264",
        #             str(viz_dir / "matches.mp4"),
        #         ]
        #     )
        #     wandb.log(
        #         {"Matches": wandb.Video(str(viz_dir / "matches.mp4"), format="mp4")}
        #     )
        #     for img in viz_dir.glob("*.png"):
        #         img.unlink()
