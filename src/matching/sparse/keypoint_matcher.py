import multiprocessing as mp
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from jaxtyping import Float, Int
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


class KeypointMatcher:
    def __init__(
        self,
        cfg: KeypointMatcherCfg,
        logger: wandb.sdk.wandb_run.Run,
        save_dir: Path,
    ):
        self.cfg = cfg
        self.logger = logger
        self.save_dir = save_dir

        self.matcher_params = {
            "width_confidence": -1,
            "depth_confidence": -1,
            "mp": True,
        }

    def _chunkify(
        self,
        pairs: list[tuple[int, int]],
        n_cpu: int,
    ) -> list[list[tuple[int, int]]]:
        """Splits an pairs into n_cpu chunks."""
        print(f"Chunking pairs into {n_cpu} chunks.")
        chunk_size = len(pairs) // n_cpu + 1
        return [pairs[i : i + chunk_size] for i in range(0, len(pairs), chunk_size)]

    def match_keypoints(
        self,
        paths: list[Path],
        feature_dir: Path,
        index_pairs: list[tuple[int, int]],
        fixed: bool = False,
    ) -> None:
        """Computes distances between keypoints of images.

        Stores output at feature_dir/matches.h5
        """
        if (feature_dir / "matches_0.h5").exists():
            return

        if self.cfg.mask:
            mask_imgs = [
                cv2.imread(
                    Path(str(path.parent).replace("images", "masks")) / path.name,
                    cv2.IMREAD_GRAYSCALE,
                )
                for path in paths
            ]

        n_cpu = min(mp.cpu_count(), 20)
        index_pairs_chunks = self._chunkify(index_pairs, n_cpu)

        futures = []
        with ProcessPoolExecutor() as executor:
            for i_proc, sub_index_pairs in enumerate(index_pairs_chunks):
                future = executor.submit(
                    self._keypoint_distances,
                    sub_index_pairs,
                    paths,
                    mask_imgs if self.cfg.mask else None,
                    feature_dir,
                    i_proc,
                    fixed,
                )
                futures.append(future)
                print(f"Chunk {i_proc + 1}/{len(index_pairs_chunks)} submitted.")
            _ = [f.result() for f in futures]

    def _keypoint_distances(
        self,
        index_pairs: list[tuple[int, int]],
        paths: list[Path],
        mask_imgs: list[torch.Tensor] | None,
        feature_dir: Path,
        i_proc: int,
        fixed: bool,
    ):
        gpu_id = 0 if i_proc % 2 else 1
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

        _matcher = KF.LightGlueMatcher("aliked", self.matcher_params).eval().to(device)

        with h5py.File(
            feature_dir / "keypoints.h5", mode="r"
        ) as f_keypoints, h5py.File(
            feature_dir / "descriptors.h5", mode="r"
        ) as f_descriptors, h5py.File(
            (feature_dir / f"matches_{i_proc}.h5"), mode="w"
        ) as f_matches:
            for idx1, idx2 in tqdm(
                index_pairs, desc=f"Matching keypoints in the process {i_proc}"
            ):
                key1 = "-".join(paths[idx1].parts[-3:])
                key2 = "-".join(paths[idx2].parts[-3:])

                keypoints1 = torch.from_numpy(f_keypoints[key1][...]).to(device)
                keypoints2 = torch.from_numpy(f_keypoints[key2][...]).to(device)
                descriptors1 = torch.from_numpy(f_descriptors[key1][...]).to(device)
                descriptors2 = torch.from_numpy(f_descriptors[key2][...]).to(device)

                with torch.inference_mode():
                    _, indices = _matcher(
                        descriptors1,
                        descriptors2,
                        KF.laf_from_center_scale_ori(keypoints1[None]),
                        KF.laf_from_center_scale_ori(keypoints2[None]),
                    )

                # If mask is enabled, remove the matches that are in the mask
                if self.cfg.mask:
                    mask_img1 = mask_imgs[idx1].to(device)

                    # Get the pixel positions of the matches
                    matched_keypoints1 = keypoints1[indices[:, 0], :2]

                    mask1 = mask_img1[
                        matched_keypoints1[:, 1].long(), matched_keypoints1[:, 0].long()
                    ]
                    masked_matched_keypoints1 = matched_keypoints1[mask1 == 0]
                    indices1 = torch.nonzero(
                        torch.isin(keypoints1, masked_matched_keypoints1), as_tuple=True
                    )[0].unique()
                    indices = indices[
                        torch.nonzero(
                            torch.isin(indices.reshape(2, -1)[0], indices1),
                            as_tuple=True,
                        )[0].unique(),
                        :,
                    ]

                # We have matches to consider
                n_matches = len(indices)
                if n_matches:
                    if self.cfg.verbose:
                        print(f"{key1}-{key2}: {n_matches} matches")

                    # Store the matches in the group of one image
                    if n_matches >= self.cfg.min_matches:
                        if fixed:
                            key1 = "-".join(paths[0].parts[-3:])
                        group = f_matches.require_group(key1)
                        group.create_dataset(
                            key2, data=indices.detach().cpu().numpy().reshape(-1, 2)
                        )

    def match_trajectories(
        self,
        paths: list[Path],
        feature_dir: Path,
        index_pairs: list[tuple[int, int]],
        kpts_per_img: dict[
            int,
            tuple[
                Float[NDArray, "... 2"],
                Int[NDArray, "..."],
            ],
        ],
        viz: bool = False,
    ) -> None:
        """Match keypoints in the dynamic cameras exhaustively."""
        if (feature_dir / "matches.h5").exists():
            print("\t Trajectories are already matched, skipping.")
            return

        if self.cfg.mask:
            mask_imgs = [
                cv2.imread(
                    Path(str(path.parent).replace("images", "masks")) / path.name,
                    cv2.IMREAD_GRAYSCALE,
                )
                for path in paths
            ]

        if viz:
            viz_dir = self.save_dir / "matches_viz"
            viz_dir.mkdir(parents=True, exist_ok=True)
            images = [load_image(path) for path in paths]

        with h5py.File(feature_dir / "matches.h5", mode="w") as f_matches:
            for idx1, idx2 in tqdm(
                index_pairs, desc="Matching keypoints in dynamic cameras"
            ):
                key1 = "-".join(paths[idx1].parts[-3:])
                key2 = "-".join(paths[idx2].parts[-3:])

                kpts1, traj_ids1 = kpts_per_img[idx1]
                kpts2, traj_ids2 = kpts_per_img[idx2]
                _, idx1_of_common, idx2_of_common = np.intersect1d(
                    traj_ids1, traj_ids2, assume_unique=True, return_indices=True
                )
                indices = np.stack([idx1_of_common, idx2_of_common], axis=1)

                # Mask out keypoints that are in a dynamic region
                if self.cfg.mask:
                    mask1 = mask_imgs[idx1]
                    matched_kpts1 = kpts1[indices[:, 0]]
                    mask_val = mask1[
                        matched_kpts1[:, 1].astype(int), matched_kpts1[:, 0].astype(int)
                    ]
                    indices = indices[mask_val == 0].copy()

                if viz:
                    image1 = images[idx1]
                    image2 = images[idx2]
                    viz2d.plot_images([image1, image2])
                    viz2d.plot_matches(
                        kpts1[indices[:, 0]], kpts2[indices[:, 1]], color="lime", lw=0.2
                    )
                    viz2d.add_text(0, f"{key1}_{key2}", fs=20)
                    viz2d.save_plot(viz_dir / f"{key1}_{key2}.png")
                    plt.close()

                if len(indices) >= self.cfg.min_matches:
                    group = f_matches.require_group(key1)
                    group.create_dataset(key2, data=indices)

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
                    str(viz_dir / "matches.mp4"),
                ]
            )

    def match_keypoints_traj(
        self,
        paths: list[Path],
        kpts_per_img: dict[
            int,
            tuple[
                Float[NDArray, "... 2"],
                Float[NDArray, "... D"],
                Int[NDArray, "..."],
                Int[NDArray, "..."],
            ],
        ],
        index_pairs: list[tuple[int, int]],
    ) -> dict[int, dict[int, tuple[np.ndarray, np.ndarray]]]:
        if self.cfg.mask:
            mask_imgs = [
                cv2.imread(
                    Path(str(path.parent).replace("images", "masks")) / path.name,
                    cv2.IMREAD_GRAYSCALE,
                )
                for path in paths
            ]

        n_cpu = min(mp.cpu_count(), 4)
        index_pairs_chunks = self._chunkify(index_pairs, n_cpu)

        futures = []
        with ProcessPoolExecutor() as executor:
            for i_proc, sub_index_pairs in enumerate(index_pairs_chunks):
                future = executor.submit(
                    self._keypoint_distances_traj,
                    sub_index_pairs,
                    mask_imgs if self.cfg.mask else None,
                    kpts_per_img,
                    i_proc,
                )
                futures.append(future)
                print(f"Chunk {i_proc + 1}/{len(index_pairs_chunks)} submitted.")
            result = [f.result() for f in futures]
        # result = self._keypoint_distances_traj(
        #     index_pairs,
        #     mask_imgs if self.cfg.mask else None,
        #     kpts_per_img,
        #     0,
        # )

        matched_traj_ids = defaultdict(dict)
        for sub_result in result:
            matched_traj_ids.update(sub_result)
        return matched_traj_ids

    def _keypoint_distances_traj(
        self,
        index_pairs: list[tuple[int, int]],
        mask_imgs: list[torch.Tensor] | None,
        kpts_per_img: dict[
            int,
            tuple[
                Float[NDArray, "... 2"],
                Float[NDArray, "... D"],
                Int[NDArray, "..."],
                Int[NDArray, "..."],
            ],
        ],
        i_proc: int,
    ):
        gpu_id = 0 if i_proc % 2 else 1
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

        _matcher = KF.LightGlueMatcher("aliked", self.matcher_params).eval().to(device)

        matched_traj_ids = defaultdict(dict)
        for idx1, idx2 in tqdm(
            index_pairs, desc=f"Matching keypoints in the process {i_proc}"
        ):
            if idx1 not in kpts_per_img or idx2 not in kpts_per_img:
                continue
            kpts1, desc1, traj_ids1, idx_in_traj_list1 = kpts_per_img[idx1]
            kpts2, desc2, traj_ids2, idx_in_traj_list2 = kpts_per_img[idx2]

            keypoints1 = torch.from_numpy(kpts1).to(device)
            keypoints2 = torch.from_numpy(kpts2).to(device)
            descriptors1 = torch.from_numpy(desc1).to(device)
            descriptors2 = torch.from_numpy(desc2).to(device)

            with torch.inference_mode():
                _, indices = _matcher(
                    descriptors1,
                    descriptors2,
                    KF.laf_from_center_scale_ori(keypoints1[None]),
                    KF.laf_from_center_scale_ori(keypoints2[None]),
                )

            # If mask is enabled, remove the matches that are in the mask
            if self.cfg.mask:
                mask_img1 = mask_imgs[idx1].to(device)

                # Get the pixel positions of the matches
                matched_keypoints1 = keypoints1[indices[:, 0], :2]

                mask1 = mask_img1[
                    matched_keypoints1[:, 1].long(), matched_keypoints1[:, 0].long()
                ]
                masked_matched_keypoints1 = matched_keypoints1[mask1 == 0]
                indices1 = torch.nonzero(
                    torch.isin(keypoints1, masked_matched_keypoints1), as_tuple=True
                )[0].unique()
                indices = indices[
                    torch.nonzero(
                        torch.isin(indices.reshape(2, -1)[0], indices1),
                        as_tuple=True,
                    )[0].unique(),
                    :,
                ]

            # We have matches to consider
            if len(indices):
                # Leave only the trajectory that starts from the frame
                indices = indices.detach().cpu().numpy().astype(int)
                indices = indices[
                    (idx_in_traj_list1[indices[:, 0]] == 0)
                    # & (idx_in_traj_list2[indices[:, 1]] == 0)
                ]
                # print(f"Filtered {len(indices)} matches after filtering by start.")

                if len(indices) >= self.cfg.min_matches:
                    # Store the matched trajectory as:  dict[frame_i][frame_j] -> (traj_id_i, traj_id_j)
                    matched_traj_ids[idx1][idx2] = (
                        traj_ids1[indices[:, 0]],
                        traj_ids2[indices[:, 1]],
                    )

        return matched_traj_ids

    def match_keypoints_fixed(
        self,
        index_pairs: list[tuple[int, int]],
        paths: list[Path],
        feature_dir: Path,
        viz: bool = False,
    ):
        """Match keypoints in the fixed camera exhaustively."""
        if (feature_dir / "matches_fixed.h5").exists():
            return

        # gpu_id = 0 if i_proc % 2 else 1
        gpu_id = 0
        _device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

        _matcher = KF.LightGlueMatcher("aliked", self.matcher_params).eval().to(_device)

        if self.cfg.mask:
            mask_imgs = [
                cv2.imread(
                    Path(str(path.parent).replace("images", "masks")) / path.name,
                    cv2.IMREAD_GRAYSCALE,
                )
                for path in paths
            ]

        if viz:
            viz_dir = self.save_dir / "matches_fixed_viz"
            viz_dir.mkdir(parents=True, exist_ok=True)

        # Match keypoints in the fixed camera over all pairs
        indices_all = []
        kpts1_all = []
        key2_all = []
        with h5py.File(
            feature_dir / "keypoints.h5", mode="r+"
        ) as f_keypoints, h5py.File(
            feature_dir / "descriptors.h5", mode="r"
        ) as f_descriptors, h5py.File(
            (feature_dir / "matches_fixed.h5"), mode="w"
        ) as f_matches:
            for idx1, idx2 in tqdm(
                index_pairs, desc="Matching keypoints in the fixed camera"
            ):
                key1 = "-".join(paths[idx1].parts[-3:])
                key2 = "-".join(paths[idx2].parts[-3:])

                keypoints1 = torch.from_numpy(f_keypoints[key1][...]).to(_device)
                keypoints2 = torch.from_numpy(f_keypoints[key2][...]).to(_device)
                descriptors1 = torch.from_numpy(f_descriptors[key1][...]).to(_device)
                descriptors2 = torch.from_numpy(f_descriptors[key2][...]).to(_device)

                with torch.inference_mode():
                    _, indices = _matcher(
                        descriptors1,
                        descriptors2,
                        KF.laf_from_center_scale_ori(keypoints1[None]),
                        KF.laf_from_center_scale_ori(keypoints2[None]),
                    )

                # If mask is enabled, remove the matches that are in the mask
                if self.cfg.mask:
                    mask_img1 = mask_imgs[idx1].to(_device)

                    # Get the pixel positions of the matches
                    matched_keypoints1 = keypoints1[indices[:, 0], :2]

                    mask1 = mask_img1[
                        matched_keypoints1[:, 1].long(), matched_keypoints1[:, 0].long()
                    ]
                    masked_matched_keypoints1 = matched_keypoints1[mask1 == 0]
                    indices1 = torch.nonzero(
                        torch.isin(keypoints1, masked_matched_keypoints1), as_tuple=True
                    )[0].unique()
                    indices = indices[
                        torch.nonzero(
                            torch.isin(indices.reshape(2, -1)[0], indices1),
                            as_tuple=True,
                        )[0].unique(),
                        :,
                    ]

                indices_all.append(indices.detach().cpu().numpy())
                kpts1_all.append(keypoints1.detach().cpu().numpy())
                key2_all.append(key2)

            kpts1_all_stacked = np.vstack(kpts1_all.copy())  # shape (N_total, 2)
            kpts1_unique, inverse_indices = np.unique(
                kpts1_all_stacked, axis=0, return_inverse=True
            )
            print("Before unique", kpts1_all_stacked.shape)
            print("After unique", kpts1_unique.shape)
            key1 = "-".join(paths[0].parts[-3:]) + "_unique"
            f_keypoints[key1] = kpts1_unique
            start_idx = 0
            for kpts1, indices, key2 in zip(kpts1_all, indices_all, key2_all):
                idx1_global = indices[:, 0] + start_idx
                indices[:, 0] = inverse_indices[idx1_global]
                start_idx += len(kpts1)

                # We have matches to consider
                n_matches = len(indices)
                # Store the matches in the group of one image
                if n_matches >= self.cfg.min_matches:
                    group = f_matches.require_group(key1)
                    group.create_dataset(key2, data=indices.reshape(-1, 2))
