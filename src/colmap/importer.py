from pathlib import Path
from typing import Literal

import h5py
import wandb
from tqdm import tqdm
from tqdm.contrib import tenumerate

from src.colmap.database import COLMAPDatabase
from src.colmap.h5_to_db import add_keypoints, add_matches, create_camera


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
            added = set()
            for match_file in feature_dir.glob("matches_*.h5"):
                added = add_matches(
                    db,
                    match_file,
                    fname_to_id,
                    added,
                )

        elif matching_type == "tracking":
            n_frames = len(image_paths) // 4
            print(f"Importing {n_frames * 4} frames into COLMAP database...")
            fname_to_id = {}
            for i, pth in tenumerate(image_paths, desc="Importing images"):
                # if i < n_frames:
                #     continue
                if i % n_frames == 0:
                    camera_id = create_camera(db, pth, "simple-pinhole")
                img_path = "/".join(pth.parts[-3:])
                image_id = db.add_image(name=img_path, camera_id=camera_id)
                fname_to_id[str(i)] = image_id
            # camera_id = create_camera(db, image_paths[0], "simple-pinhole")
            # image_id = db.add_image(name="/".join(image_paths[0].parts[-3:]), camera_id=camera_id)
            # fname_to_id["0"] = image_id

            keypoint_f = h5py.File((feature_dir / "keypoints.h5"), "r")

            print("Importing keypoints into the database...")
            for key in tqdm(list(keypoint_f.keys())):
                keypoints = keypoint_f[key][()]
                db.add_keypoints(fname_to_id[key], keypoints)

            print("Importing matches into the database...")
            added = set()
            for match_file in feature_dir.glob("matches_*.h5"):
                added = add_matches(
                    db,
                    match_file,
                    fname_to_id,
                    added,
                )

        db.commit()
