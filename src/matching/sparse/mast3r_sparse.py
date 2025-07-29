from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.cuda.amp
from jaxtyping import Float
from numpy.typing import NDArray
from PIL import Image

from src.submodules.mast3r.dust3r.dust3r.utils.image import (
    ImgNorm,
    _resize_pil_image,
    exif_transpose,
    heif_support_enabled,
)
from src.submodules.mast3r.mast3r.fast_nn import bruteforce_reciprocal_nns
from src.submodules.mast3r.mast3r.model import AsymmetricMASt3R


class MASt3RSparseMatcher:
    def __init__(
        self,
        device: torch.device,
        model_path: Path,
        filter_threshold: float,
    ):
        self.device = device
        self.filter_threshold = filter_threshold

        self.model = AsymmetricMASt3R.from_pretrained(model_path).eval().to(device)

    @torch.inference_mode()
    def __call__(
        self,
        pth1: Path,
        pth2: Path,
        kpts1: Float[NDArray, "... 2"],
        kpts2: Float[NDArray, "... 2"],
    ):
        """Loads images, resizes and crops them and returns a list of dictionaries
        containing the processed image and related metadata."""

        origin_kpts1 = kpts1.copy()
        origin_kpts2 = kpts2.copy()
        orig_img1 = cv2.imread(str(pth1))
        orig_img2 = cv2.imread(str(pth2))
        orig_H1, orig_W1 = orig_img1.shape[:2]
        orig_H2, orig_W2 = orig_img2.shape[:2]

        # if cropper:
        #     orig_img1 = cv2.cvtColor(orig_img1, cv2.COLOR_BGR2RGB)
        #     orig_img2 = cv2.cvtColor(orig_img2, cv2.COLOR_BGR2RGB)
        #     cropper.set_original_image(orig_img1, orig_img2)

        # NOTE: Read images again without caches
        paired_images = self._load_images_fixed([pth1, pth2], size=512)
        crop_offset1 = paired_images[0].pop("crop_offset")
        crop_offset2 = paired_images[1].pop("crop_offset")
        size_before_crop1 = paired_images[0].pop("size_before_crop")
        size_before_crop2 = paired_images[1].pop("size_before_crop")
        cx1, cy1 = paired_images[0].pop("crop_center")
        cx2, cy2 = paired_images[1].pop("crop_center")
        halfw1, halfh1 = paired_images[0].pop("half_wh")
        halfw2, halfh2 = paired_images[1].pop("half_wh")
        img1 = paired_images[0].pop("img")
        img2 = paired_images[1].pop("img")

        # Transform keypoints to resized image coordinates
        kpts1[:, 0] = (kpts1[:, 0] / orig_W1) * size_before_crop1[0]
        kpts1[:, 1] = (kpts1[:, 1] / orig_H1) * size_before_crop1[1]
        kpts2[:, 0] = (kpts2[:, 0] / orig_W2) * size_before_crop2[0]
        kpts2[:, 1] = (kpts2[:, 1] / orig_H2) * size_before_crop2[1]

        mask1 = (
            (crop_offset1[0] <= kpts1[:, 0])
            & (kpts1[:, 0] < (cx1 + halfw1))
            & (crop_offset1[1] <= kpts1[:, 1])
            & (kpts1[:, 1] < (cy1 + halfh1))
        )
        mask2 = (
            (crop_offset2[0] <= kpts2[:, 0])
            & (kpts2[:, 0] < (cx2 + halfw2))
            & (crop_offset2[1] <= kpts2[:, 1])
            & (kpts2[:, 1] < (cy2 + halfh2))
        )

        kpts1 = kpts1[mask1] - np.array([crop_offset1[0], crop_offset1[1]])
        kpts2 = kpts2[mask2] - np.array([crop_offset2[0], crop_offset2[1]])

        with torch.autocast(self.device.type):
            shape1 = torch.tensor(img1.shape[-2:])[None].to(
                self.device, non_blocking=True
            )
            shape2 = torch.tensor(img2.shape[-2:])[None].to(
                self.device, non_blocking=True
            )
            img1 = img1.to(self.device, non_blocking=True)
            img2 = img2.to(self.device, non_blocking=True)

            # compute encoder only once
            feat1, feat2, pos1, pos2 = self.model._encode_image_pairs(
                img1, img2, shape1, shape2
            )

            # decoder 1-2
            dec1, dec2 = self.model._decoder(feat1, pos1, feat2, pos2)
            with torch.autocast(self.device.type, enabled=False):
                pred1 = self.model._downstream_head(
                    1, [tok.float() for tok in dec1], shape1
                )
                pred2 = self.model._downstream_head(
                    2, [tok.float() for tok in dec2], shape2
                )

        desc1, desc2 = (
            pred1["desc"].squeeze(0).detach(),
            pred2["desc"].squeeze(0).detach(),
        )

        conf1, conf2 = (
            pred1["desc_conf"].squeeze(0).detach(),
            pred2["desc_conf"].squeeze(0).detach(),
        )

        _, _, scores, idxs = self.nn_correspondences(
            desc1,
            desc2,
            conf1,
            conf2,
            kpts1,
            kpts2,
            origin_kpts1[mask1],
            origin_kpts2[mask2],
        )
        # NOTE
        # `idx` range is corresponding to keypoints after masking.
        # Thus, remap it to original keypoint range
        used_origin_idx1, *_ = np.where(mask1)
        used_origin_idx2, *_ = np.where(mask2)
        idxs[:, 0] = used_origin_idx1[idxs[:, 0]]
        idxs[:, 1] = used_origin_idx2[idxs[:, 1]]

        return idxs[scores >= self.filter_threshold]

    def _load_images_fixed(
        self,
        paths: list[Path],
        size: int,
        square_ok=False,
    ) -> list[dict[str, Any]]:
        """from dust3r.utils.image import load_images"""
        supported_images_extensions = [".jpg", ".jpeg", ".png"]
        if heif_support_enabled:
            supported_images_extensions += [".heic", ".heif"]
        supported_images_extensions = tuple(supported_images_extensions)

        imgs = []
        for path in paths:
            img = exif_transpose(Image.open(path)).convert("RGB")

            W1, H1 = img.size
            img = _resize_pil_image(img, size)  # resize long side to 512
            W, H = img.size
            cx, cy = W // 2, H // 2
            halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
            if not (square_ok) and W == H:
                halfh = 3 * halfw / 4
            img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
            crop_offset = (cx - halfw, cy - halfh)
            size_before_crop = (W, H)
            crop_center = (cx, cy)
            half_wh = (halfw, halfh)

            # W2, H2 = img.size
            # print(
            #     f" - adding {path} with resolution {W1}x{H1} --> {W}x{H} --> {W2}x{H2}"
            # )
            # print(crop_offset)
            imgs.append(
                dict(
                    img=ImgNorm(img)[None],  # type: ignore
                    true_shape=np.int32([img.size[::-1]]),  # type: ignore
                    idx=len(imgs),
                    instance=str(len(imgs)),
                    size_before_crop=size_before_crop,
                    crop_offset=crop_offset,
                    crop_center=crop_center,
                    half_wh=half_wh,
                )
            )

        return imgs

    def nn_correspondences(
        self,
        descs1: torch.Tensor,
        descs2: torch.Tensor,
        scores1: torch.Tensor,  # Shape(H, W)
        scores2: torch.Tensor,  # Shape(H, W)
        kpts1: np.ndarray,
        kpts2: np.ndarray,
        origin_kpts1: np.ndarray,
        origin_kpts2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        H1, W1, _ = descs1.shape
        H2, W2, _ = descs2.shape

        kpts1_tensor = torch.tensor(kpts1, dtype=torch.float32)  # Shape: (N, 2)
        descs1 = descs1.permute(2, 0, 1).unsqueeze(0)
        scores1 = scores1[None, None]
        grid1 = (
            torch.stack(
                [
                    2.0 * kpts1_tensor[:, 0] / (W1 - 1) - 1,  # Normalize x
                    2.0 * kpts1_tensor[:, 1] / (H1 - 1) - 1,  # Normalize y
                ],
                dim=-1,
            )
            .unsqueeze(0)
            .unsqueeze(2)
            .to(descs1.device)
        )
        descs1 = torch.nn.functional.grid_sample(
            descs1, grid1, align_corners=True, mode="bilinear"
        )  # Shape: (1, C, N, 1)
        scores1 = torch.nn.functional.grid_sample(
            scores1, grid1, align_corners=True, mode="bilinear"
        )  # Shape: (1, 1, N, 1)
        descs1 = descs1.squeeze(0).squeeze(-1).T  # Shape: (N, C)
        scores1 = scores1.squeeze()  # Shape: (N)

        kpts2_tensor = torch.tensor(kpts2, dtype=torch.float32)  # Shape: (N, 2)
        descs2 = descs2.permute(2, 0, 1).unsqueeze(0)
        scores2 = scores2[None, None]
        grid2 = (
            torch.stack(
                [
                    2.0 * kpts2_tensor[:, 0] / (W2 - 1) - 1,  # Normalize x
                    2.0 * kpts2_tensor[:, 1] / (H2 - 1) - 1,  # Normalize y
                ],
                dim=-1,
            )
            .unsqueeze(0)
            .unsqueeze(2)
            .to(descs2.device)
        )
        descs2 = torch.nn.functional.grid_sample(
            descs2, grid2, align_corners=True, mode="bilinear"
        )  # Shape: (1, C, N, 1)
        scores2 = torch.nn.functional.grid_sample(
            scores2, grid2, align_corners=True, mode="bilinear"
        )  # Shape: (1, 1, N, 1)
        descs2 = descs2.squeeze(0).squeeze(-1).T  # Shape: (N, C)
        scores2 = scores2.squeeze()  # Shape: (N)

        nn1, nn2 = bruteforce_reciprocal_nns(
            descs1,
            descs2,
            device=descs1.device,  # type: ignore
            dist="dot",
            block_size=2**13,
        )
        reciprocal_in_P1 = nn2[nn1] == np.arange(len(nn1))

        scores = scores2[nn1][reciprocal_in_P1].cpu().numpy()
        mkpts1 = origin_kpts1[reciprocal_in_P1]
        mkpts2 = origin_kpts2[nn1][reciprocal_in_P1]

        idx1, *_ = np.where(reciprocal_in_P1)
        idx2 = nn1[reciprocal_in_P1]

        assert len(idx1) == len(idx2)
        idx = np.concatenate([idx1[..., None], idx2[..., None]], axis=1)

        return mkpts1, mkpts2, scores, idx
