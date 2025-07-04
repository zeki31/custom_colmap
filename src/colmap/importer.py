from pathlib import Path
from typing import Literal

import numpy as np
import wandb

from src.colmap.database import COLMAPDatabase
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
            add_matches(
                db,
                feature_dir,
                fname_to_id,
            )

        elif matching_type == "tracking":
            # 1. Add a camera (or cameras)
            camera_id = create_camera(db, image_paths[0], "simple-pinhole")
            # 2. Add images
            for pth in image_paths:
                img_path = "/".join(pth.parts[-3:])
                db.add_image(name=img_path, camera_id=camera_id)

            image_ids = {}
            for name, image_id in db.execute("SELECT name, image_id FROM images;"):
                image_ids[name] = image_id

            colmap_feat_match_data = traj_to_matches(image_paths, feature_dir)

            print("Importing keypoints into the database...")
            for image_name, image_id in image_ids.items():
                keypoints = np.array(colmap_feat_match_data[image_name].keypoints)
                keypoints += 0.5  # COLMAP origin
                if keypoints.shape[0] == 0:
                    print(f"Warning: No keypoints for image {image_name}, skipping.")
                    continue

                db.add_keypoints(image_id, keypoints)

            print("Importing matches into the database...")
            matched = set()
            for image_name, image_id in image_ids.items():
                matches = colmap_feat_match_data[image_name].match_pairs
                for pair, match in matches.items():
                    # get the image name and then id
                    name0, name1 = pair.split("-")
                    id0, id1 = image_ids[name0], image_ids[name1]
                    if len({(id0, id1), (id1, id0)} & matched) > 0:
                        continue
                    match = np.array(match)
                    db.add_matches(id0, id1, match)
                    matched |= {(id0, id1), (id1, id0)}

        db.commit()
