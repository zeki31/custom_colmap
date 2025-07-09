import warnings
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import wandb
from tqdm import tqdm

from src.colmap.database import COLMAPDatabase, image_ids_to_pair_id
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
            print(f"Importing {len(image_paths)} images into COLMAP database...")
            fname_to_id = {}
            for i in tqdm(range(n_frames, len(image_paths)), desc="Importing images"):
                pth = image_paths[i]
                if i % n_frames == 0:
                    camera_id = create_camera(db, pth, "simple-pinhole")
                key = "-".join(pth.parts[-3:])
                img_path = key.replace("-", "/")
                image_id = db.add_image(name=img_path, camera_id=camera_id)
                fname_to_id[key] = image_id
            # Handle the fixed camera separately
            camera_id = create_camera(db, image_paths[0], "simple-pinhole")
            key = "-".join(image_paths[0].parts[-3:])
            img_path = key.replace("-", "/")
            image_id = db.add_image(name=img_path, camera_id=camera_id)
            fname_to_id[key] = image_id

            keypoint_f = h5py.File((feature_dir / "keypoints.h5"), "r")

            print("Importing keypoints into the database...")
            for key in tqdm(list(keypoint_f.keys())):
                if "1_fixed" in key:
                    continue
                keypoints = keypoint_f[key][()]
                db.add_keypoints(fname_to_id[key], keypoints)
            # Handle the fixed camera keypoints separately
            key_fixed = "-".join(image_paths[0].parts[-3:])
            keypoints_fixed = keypoint_f[key_fixed][()]
            db.add_keypoints(fname_to_id[key_fixed], keypoints_fixed)

            print("Importing matches into the database...")
            added = set()
            # Matches from CoTracker3
            trajectories = np.load(feature_dir / "track.npy", allow_pickle=True).item()
            index_pairs = self.retriever.get_index_pairs(image_paths, "frame")
            for idx1, idx2 in tqdm(index_pairs, desc="Importing matches from tracks"):
                traj_ids_1 = list(trajectories.invert_maps[idx1].keys())
                traj_ids_2 = list(trajectories.invert_maps[idx2].keys())
                shared_traj_ids = set(traj_ids_1) & set(traj_ids_2)
                indices = np.stack(
                    [
                        [traj_ids_1.index(traj_id) for traj_id in shared_traj_ids],
                        [traj_ids_2.index(traj_id) for traj_id in shared_traj_ids],
                    ],
                    axis=1,
                )

                key1 = "-".join(image_paths[idx1].parts[-3:])
                key2 = "-".join(image_paths[idx2].parts[-3:])
                id_1 = fname_to_id[key1]
                id_2 = fname_to_id[key2]
                pair_id = image_ids_to_pair_id(id_1, id_2)
                if pair_id in added:
                    warnings.warn(f"Pair {pair_id} ({id_1}, {id_2}) already added!")
                    continue

                db.add_matches(id_1, id_2, indices)

                added.add(pair_id)

            # # Matches from LightGlue
            # for match_file in feature_dir.glob("matches*.h5"):
            #     added = add_matches(
            #         db,
            #         match_file,
            #         fname_to_id,
            #         added,
            #     )

        db.commit()
