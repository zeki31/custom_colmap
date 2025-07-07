# ParticleSfM
# Copyright (C) 2022  ByteDance Inc.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


class imageMatchData:
    def __init__(self, image_id: int):
        self.image_id = image_id
        self.keypoints = set()
        self.match_pairs = {}

    def insert_keypoint(self, kp1):
        # kp1: [x,y]
        self.keypoints.add(kp1)

    def insert_match(self, tgt_img_id, kp_ind1, kp_ind2):
        # kp_ind1: the keypoint index in current image
        # kp_ind2: the keypoint index in target matched image (refered by tgt_img_id)
        match_key = str(self.image_id) + "-" + str(tgt_img_id)
        if match_key in self.match_pairs.keys():
            self.match_pairs[match_key].add((kp_ind1, kp_ind2))
        else:
            self.match_pairs[match_key] = set(((kp_ind1, kp_ind2),))

    def rename_matches(self, image_names: list[Path]):
        # rename the matching pairs, replace the sorted id to image names
        old_keys = np.copy(list(self.match_pairs.keys()))
        for key in old_keys:
            self_id, tgt_img_id = key.split("-")
            self_id, tgt_img_id = int(self_id), int(tgt_img_id)
            assert self_id == self.image_id

            self_name = "/".join(image_names[self_id].parts[-3:])
            tgt_name = "/".join(image_names[tgt_img_id].parts[-3:])
            new_key = self_name + "-" + tgt_name
            self.match_pairs[new_key] = self.match_pairs.pop(key)


def traj_to_matches(image_paths: list[Path], feature_dir: Path):
    if (feature_dir / "colmap_datas.npy").exists():
        colmap_datas = np.load(
            feature_dir / "colmap_datas.npy", allow_pickle=True
        ).item()
        if not isinstance(colmap_datas, dict):
            colmap_datas = colmap_datas.as_dict()
        return colmap_datas

    trajectories = np.load(feature_dir / "track.npy", allow_pickle=True).item()
    if not isinstance(trajectories, dict):
        trajectories = trajectories.as_dict()

    # initialize each image
    image_datas = []
    for i in range(len(image_paths)):
        image_datas.append(imageMatchData(image_id=i))

    mask_dir = Path(str(image_paths[0].parent).replace("images", "masks"))
    mask_imgs = [
        cv2.imread(mask_dir / pth.name, cv2.IMREAD_GRAYSCALE) for pth in image_paths
    ]

    for traj in tqdm(trajectories.values(), desc="Processing trajectories"):
        locations, frame_ids = np.array(traj["locations"]), np.array(traj["frame_ids"])
        assert locations.shape[0] == frame_ids.shape[0]

        # Obtain all the keypoint ids in each image
        img_ids, kp_inds = [], []
        for location, img_id in zip(locations, frame_ids):
            mx, my = location

            if mask_imgs[img_id][int(my), int(mx)]:
                continue

            mx, my, img_id = float(mx), float(my), int(img_id)
            if img_id < len(image_paths) // 4:
                img_id = 0
            image_datas[img_id].insert_keypoint([mx, my])
            ind = len(image_datas[img_id].keypoints) - 1
            kp_inds.append(ind)
            img_ids.append(img_id)

        for j in range(len(img_ids)):
            for k in range(len(img_ids)):
                if k == j:
                    continue
                image_datas[img_ids[j]].insert_match(img_ids[k], kp_inds[j], kp_inds[k])

    # Read the images and project the trajectory-based image-id to image name
    # The colmap does not necessarily follow the sorted image name as ids
    colmap_datas = {}
    for i, img_name in enumerate(image_paths):
        image_datas[i].rename_matches(image_paths)
        img_name = "/".join(img_name.parts[-3:])
        colmap_datas[img_name] = image_datas[i]

    np.save(feature_dir / "colmap_datas.npy", colmap_datas)

    return colmap_datas
