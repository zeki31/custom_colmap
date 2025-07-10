from pathlib import Path
from typing import Literal

import h5py
import wandb

from src.colmap.database import COLMAPDatabase
from src.colmap.h5_to_db import add_keypoints, add_matches, create_camera
from src.matching.retriever import Retriever


class COLMAPImporter:
    def __init__(self, logger: wandb.sdk.wandb_run.Run, retriever: Retriever):
        self.logger = logger
        self.retriever = retriever

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
            print("Importing keypoints into the database...")
            fname_to_id = add_keypoints(db, feature_dir, base_dir, "simple-pinhole")
            # Handle the fixed camera separately
            camera_id = create_camera(db, image_paths[0], "simple-pinhole")
            key_fixed = "-".join(image_paths[0].parts[-3:]) + "_unique"
            img_path = "-".join(image_paths[0].parts[-3:]).replace("-", "/")
            image_id = db.add_image(name=img_path, camera_id=camera_id)
            fname_to_id[key_fixed] = image_id
            keypoint_f = h5py.File((feature_dir / "keypoints.h5"), "r")
            keypoints_fixed = keypoint_f[key_fixed][()]
            db.add_keypoints(fname_to_id[key_fixed], keypoints_fixed)

            print("Importing matches into the database...")
            added = set()
            for match_file in feature_dir.glob("matches_*.h5"):
                added = add_matches(
                    db,
                    match_file,
                    fname_to_id,
                    added,
                )

        elif matching_type == "tracking":
            fname_to_id = add_keypoints(db, feature_dir, base_dir, "simple-pinhole")
            # # Handle the fixed camera separately
            # camera_id = create_camera(db, image_paths[0], "simple-pinhole")
            # key_fixed = "-".join(image_paths[0].parts[-3:]) + "_unique"
            # img_path = "-".join(image_paths[0].parts[-3:]).replace("-", "/")
            # image_id = db.add_image(name=img_path, camera_id=camera_id)
            # fname_to_id[key_fixed] = image_id
            # keypoint_f = h5py.File((feature_dir / "keypoints.h5"), "r")
            # keypoints_fixed = keypoint_f[key_fixed][()]
            # db.add_keypoints(fname_to_id[key_fixed], keypoints_fixed)

            print("Importing matches into the database...")
            added = set()
            for match_file in feature_dir.glob("matches*.h5"):
                added = add_matches(
                    db,
                    match_file,
                    fname_to_id,
                    added,
                )

        db.commit()
