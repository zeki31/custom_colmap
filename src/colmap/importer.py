import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import wandb
from tqdm.contrib import tenumerate

from src.colmap.database import COLMAPDatabase, image_ids_to_pair_id
from src.colmap.h5_to_db import (
    add_fixed_kpts_matches,
    add_keypoints,
    add_matches,
    create_camera,
)
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
            added = set()
            for match_file in feature_dir.glob("matches_*.h5"):
                if "fixed" in match_file.name:
                    continue
                added = add_matches(
                    db,
                    match_file,
                    fname_to_id,
                    added,
                )
            add_fixed_kpts_matches(
                db,
                (feature_dir / "matches_fixed.h5"),
                fname_to_id,
                image_paths[0],
                "simple-pinhole",
            )

        elif matching_type == "tracking":
            n_frames = len(image_paths)
            print(f"Importing {n_frames} frames into COLMAP database...")
            for i, pth in tenumerate(image_paths, desc="Importing images"):
                if i % n_frames == 0:
                    camera_id = create_camera(db, pth, "simple-pinhole")
                img_path = "/".join(pth.parts[-3:])
                db.add_image(name=img_path, camera_id=camera_id)

            colmap_feat_match_data = traj_to_matches(image_paths, feature_dir)

            image_ids = {}
            for name, image_id in db.execute("SELECT name, image_id FROM images;"):
                image_ids[name] = image_id

            print("Importing keypoints into the database...")
            for image_name, image_id in image_ids.items():
                keypoints = np.array(colmap_feat_match_data[image_name].keypoints)
                # keypoints += 0.5  # COLMAP origin

                db.add_keypoints(image_id, keypoints)
                keypoint_f = h5py.File(os.path.join(h5_path, "keypoints.h5"), "r")

                camera_id = None
                fname_to_id = {}
                for key in tqdm(list(keypoint_f.keys())):
                    keypoints = keypoint_f[key][()]
                    if "fixed" in key:
                        continue  # skip fixed keypoints

                    filename = key.replace("-", "/")
                    path = image_path / filename
                    if not path.is_file():
                        raise IOError(f"Invalid image path {path}")

                    if camera_id is None:
                        camera_id = create_camera(db, path, camera_model)
                    image_id = db.add_image(filename, camera_id)
                    fname_to_id[key] = image_id

                    db.add_keypoints(image_id, keypoints)

                return fname_to_id

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

        db.commit()
