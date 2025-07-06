import warnings
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import wandb

from src.colmap.database import COLMAPDatabase, image_ids_to_pair_id
from src.colmap.h5_to_db import add_keypoints, add_matches, create_camera
from src.colmap.traj2matches import traj_to_matches


class COLMAPImporter:
    def __init__(
        self,
        logger: wandb.sdk.wandb_run.Run,
    ):
        self.logger = logger

    def import_into_colmap(
        self,
        database_path: Path,
        feature_dir: Path,
        image_paths: list[Path],
        matching_type: Literal["sparse", "tracking"],
        base_dir: Path,
    ) -> None:
        """Adds keypoints into colmap"""
        db = COLMAPDatabase.connect(database_path)
        db.create_tables()

        if matching_type == "sparse":
            fname_to_id = add_keypoints(db, feature_dir, base_dir, "simple-pinhole")
            for match_file in feature_dir.glob("matches_*.h5"):
                add_matches(
                    db,
                    match_file,
                    fname_to_id,
                )

        elif matching_type == "tracking":
            colmap_feat_match_data = {}
            for cam_name in ["2_dynA", "3_dynB"]:
                sub_feature_dir = feature_dir / cam_name
                sub_image_paths = [
                    image_path
                    for image_path in image_paths
                    if cam_name in str(image_path)
                ]

                camera_id = create_camera(db, sub_image_paths[0], "simple-pinhole")
                for pth in sub_image_paths:
                    img_path = "/".join(pth.parts[-3:])
                    db.add_image(name=img_path, camera_id=camera_id)

                sub_colmap_feat_match_data = traj_to_matches(
                    sub_image_paths, sub_feature_dir
                )
                colmap_feat_match_data.update(sub_colmap_feat_match_data)

            image_ids = {}
            for name, image_id in db.execute("SELECT name, image_id FROM images;"):
                image_ids[name] = image_id

            print("Importing keypoints into the database...")
            keypoint_f = h5py.File(feature_dir / "keypoints.h5", "r")
            for image_name, image_id in image_ids.items():
                keypoints = np.array(colmap_feat_match_data[image_name].keypoints)
                # keypoints += 0.5  # COLMAP origin
                keypoints = np.concatenate(
                    ([keypoints, keypoint_f[image_name.replace("/", "-")][()]]), axis=0
                )
                keypoints = np.unique(keypoints, axis=0)

                db.add_keypoints(image_id, keypoints)

            print("Importing matches into the database...")
            added = set()
            for image_name, image_id in image_ids.items():
                matches = colmap_feat_match_data[image_name].match_pairs
                for pair, match in matches.items():
                    # get the image name and then id
                    name_1, name_2 = pair.split("-")
                    id_1, id_2 = image_ids[name_1], image_ids[name_2]

                    pair_id = image_ids_to_pair_id(id_1, id_2)
                    if pair_id in added:
                        warnings.warn(f"Pair ({name_1}, {name_2}) already added!")
                        continue
                    added.add(pair_id)

                    db.add_matches(id_1, id_2, np.array(match))

            match_file = h5py.File(feature_dir / "matches.h5", "r")
            for key_1 in match_file.keys():
                group = match_file[key_1]
                id_1 = image_ids[key_1.replace("-", "/")]
                for key_2 in group.keys():
                    id_2 = image_ids[key_2.replace("-", "/")]

                    pair_id = image_ids_to_pair_id(id_1, id_2)
                    if pair_id in added:
                        warnings.warn(
                            f"Pair ({image_name}, {key_2.replace('-', '/')}) already added!"
                        )
                        continue

                    matches = group[key_2][()]
                    db.add_matches(id_1, id_2, matches)

                    added.add(pair_id)

        db.commit()
