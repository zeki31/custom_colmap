from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import kornia.feature as KF
import torch
from tqdm import tqdm


@dataclass
class KeypointMatcherCfg:
    min_matches: int = 15
    verbose: bool = True
    mask: bool = False


def keypoint_distances(
    paths: list[Path],
    index_pairs: list[tuple[int, int]],
    feature_dir: Path,
    cfg: KeypointMatcherCfg,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Computes distances between keypoints of images.

    Stores output at feature_dir/matches.h5
    """
    if (feature_dir / "matches.h5").exists():
        return

    if cfg.mask and (paths[0].parents[1] / "masks").exists():
        mask_dir = paths[0].parents[1] / "masks"

    matcher_params = {
        "width_confidence": -1,
        "depth_confidence": -1,
        "mp": True if "cuda" in str(device) else False,
    }
    matcher = KF.LightGlueMatcher("aliked", matcher_params).eval().to(device)

    with h5py.File(feature_dir / "keypoints.h5", mode="r") as f_keypoints, h5py.File(
        feature_dir / "descriptors.h5", mode="r"
    ) as f_descriptors, h5py.File(feature_dir / "matches.h5", mode="w") as f_matches:
        for idx1, idx2 in tqdm(index_pairs, desc="Computing keypoing distances"):
            key1 = paths[idx1].parts[-3] + "-images-" + paths[idx1].name
            key2 = paths[idx2].parts[-3] + "-images-" + paths[idx2].name

            keypoints1 = torch.from_numpy(f_keypoints[key1][...]).to(device)
            keypoints2 = torch.from_numpy(f_keypoints[key2][...]).to(device)
            descriptors1 = torch.from_numpy(f_descriptors[key1][...]).to(device)
            descriptors2 = torch.from_numpy(f_descriptors[key2][...]).to(device)

            with torch.inference_mode():
                _, indices = matcher(
                    descriptors1,
                    descriptors2,
                    KF.laf_from_center_scale_ori(keypoints1[None]),
                    KF.laf_from_center_scale_ori(keypoints2[None]),
                )

            # If mask is enabled, remove the matches that are in the mask
            if cfg.mask:
                mask_img1 = cv2.imread(mask_dir / key1, cv2.IMREAD_GRAYSCALE)
                mask_img1 = torch.from_numpy(mask_img1).to(device)

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
                        torch.isin(indices.reshape(2, -1)[0], indices1), as_tuple=True
                    )[0].unique(),
                    :,
                ]

            # We have matches to consider
            n_matches = len(indices)
            if n_matches:
                if cfg.verbose:
                    print(f"{key1}-{key2}: {n_matches} matches")

                # Store the matches in the group of one image
                if n_matches >= cfg.min_matches:
                    group = f_matches.require_group(key1)
                    group.create_dataset(
                        key2, data=indices.detach().cpu().numpy().reshape(-1, 2)
                    )
