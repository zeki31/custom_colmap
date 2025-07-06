import itertools
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import h5py
import kornia.feature as KF
import torch
import wandb
from tqdm import tqdm


@dataclass
class KeypointMatcherCfg:
    pair_generator: Literal["exhaustive", "frame-view", "view"] = "exhaustive"
    min_matches: int = 15
    verbose: bool = True
    mask: bool = False


class KeypointMatcher:
    def __init__(
        self,
        cfg: KeypointMatcherCfg,
        logger: wandb.sdk.wandb_run.Run,
    ):
        self.cfg = cfg
        self.logger = logger

        self.matcher_params = {
            "width_confidence": -1,
            "depth_confidence": -1,
            "mp": True,
        }

    def _get_index_pairs(
        self,
        paths: list[Path],
    ) -> list[tuple[int, int]]:
        if self.cfg.pair_generator == "exhaustive":
            # Obtains all possible index pairs of a list
            pairs = list(itertools.combinations(range(len(paths)), 2))
        elif self.cfg.pair_generator == "frame-view":
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
        elif self.cfg.pair_generator == "view":
            # Obtains only adjacent cameras
            # (same timestamp, different cameras)
            pairs = []
            n_frames = len(paths) // 4
            for t in range(n_frames):
                pairs.extend(
                    list(itertools.combinations(range(t, len(paths), n_frames), 2))
                )
            pairs = sorted(set(pairs))  # Remove duplicates

        return pairs

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
    ) -> None:
        """Computes distances between keypoints of images.

        Stores output at feature_dir/matches.h5
        """
        if (feature_dir / "matches_0.h5").exists():
            return

        if self.cfg.mask and (paths[0].parents[1] / "masks").exists():
            mask_dir = paths[0].parents[1] / "masks"
            mask_imgs = [
                torch.from_numpy(cv2.imread(mask_dir / path.name, cv2.IMREAD_GRAYSCALE))
                for path in paths
            ]

        index_pairs = self._get_index_pairs(paths)
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
                )
                futures.append(future)
                print(f"Chunk {i_proc + 1}/{len(index_pairs_chunks)} submitted.")
            result = [f.result() for f in futures]
        print(result)

    def _keypoint_distances(
        self,
        index_pairs: list[tuple[int, int]],
        paths: list[Path],
        mask_imgs: list[torch.Tensor] | None,
        feature_dir: Path,
        i_proc: int,
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
                        group = f_matches.require_group(key1)
                        group.create_dataset(
                            key2, data=indices.detach().cpu().numpy().reshape(-1, 2)
                        )
