import gc
import warnings
from pathlib import Path

import h5py
import numpy as np
import wandb
from PIL import ExifTags, Image
from tqdm import tqdm

from src.colmap.database import COLMAPDatabase, image_ids_to_pair_id


class COLMAPImporter:
    def __init__(
        self,
        logger: wandb.sdk.wandb_run.Run,
        base_dir: Path,
        feature_dir: Path,
        image_paths: list[Path],
        stride: int,
    ):
        self.logger = logger
        self.base_dir = base_dir
        self.feature_dir = feature_dir
        self.image_paths = image_paths
        self.stride = stride

        self.n_frames = len(image_paths) // 4
        self.camera_id = None
        self.fname_to_id = {}

    def _get_non_keyframe_keys(self, paths: list[Path]) -> list[str]:
        return [
            "-".join(pth.parts[-3:])
            for i, pth in enumerate(paths[self.n_frames :])
            if i % self.stride != 0
        ]

    def import_keyframes(self, database_path: Path) -> None:
        db = COLMAPDatabase.connect(database_path)
        db.create_tables()

        self.keyframe_keys = [
            "-".join(pth.parts[-3:])
            for pth in self.image_paths[self.n_frames :: self.stride]
        ]
        self.keyframe_keys.append("-".join(self.image_paths[0].parts[-3:]))

        self._add_keypoints(db, "simple-pinhole", self.keyframe_keys)
        self._add_keypoints_fixed(db)

        added = set()
        for match_file in self.feature_dir.glob("matches*.h5"):
            added = self._add_matches(
                db,
                match_file,
                added,
                self.keyframe_keys,
                self.keyframe_keys,
            )
        del added

        db.commit()
        db.close()
        gc.collect()

    def import_non_keyframes(self, database_path: Path) -> None:
        db = COLMAPDatabase.connect(database_path)

        non_keyframe_keys = self._get_non_keyframe_keys(self.image_paths)
        self._add_keypoints(db, "simple-pinhole", non_keyframe_keys)

        added = set()
        for match_file in self.feature_dir.glob("matches*.h5"):
            added = self._add_matches(
                db,
                match_file,
                added,
                non_keyframe_keys,
                self.keyframe_keys,
            )
        del added

        db.commit()
        db.close()
        gc.collect()

    def _add_keypoints(
        self,
        db: COLMAPDatabase,
        camera_model: str,
        include_keys: list[str] | None = None,
    ) -> None:
        with h5py.File((self.feature_dir / "keypoints.h5"), "r") as keypoint_f:
            for key in tqdm(list(keypoint_f.keys()), desc="Adding keypoints"):
                if "fixed" in key:
                    continue  # skip fixed keypoints
                if include_keys is not None and key not in include_keys:
                    continue  # skip keys not in include_keys

                keypoints = keypoint_f[key][()]

                filename = key.replace("-", "/")
                path = self.base_dir / filename
                if not path.is_file():
                    raise IOError(f"Invalid image path {path}")

                if self.camera_id is None:
                    self.camera_id = self._create_camera(db, path, camera_model)
                image_id = db.add_image(filename, self.camera_id)
                self.fname_to_id[key] = image_id

                db.add_keypoints(image_id, keypoints)

    def _add_keypoints_fixed(self, db: COLMAPDatabase):
        """Handle the fixed camera separately."""
        camera_id = self._create_camera(db, self.image_paths[0], "simple-pinhole")
        key_fixed = "-".join(self.image_paths[0].parts[-3:])
        img_path = "-".join(self.image_paths[0].parts[-3:]).replace("-", "/")
        image_id = db.add_image(name=img_path, camera_id=camera_id)
        self.fname_to_id[key_fixed] = image_id
        with h5py.File((self.feature_dir / "keypoints.h5"), "r") as keypoint_f:
            keypoints_fixed = keypoint_f[key_fixed][()]
            db.add_keypoints(self.fname_to_id[key_fixed], keypoints_fixed)

    def _add_matches(
        self,
        db: COLMAPDatabase,
        match_file: Path,
        added: set[int],
        src_include_keys: list[str] | None = None,
        dst_include_keys: list[str] | None = None,
    ) -> set[int]:
        with h5py.File(match_file, "r") as matches_f:
            for key_1 in tqdm(matches_f.keys(), desc="Adding matches"):
                if src_include_keys is not None and key_1 not in src_include_keys:
                    continue  # skip keys not in include_keys

                group = matches_f[key_1]
                for key_2 in group.keys():
                    if dst_include_keys is not None and key_2 not in dst_include_keys:
                        continue  # skip keys not in include_keys
                    id_1 = self.fname_to_id[key_1]
                    id_2 = self.fname_to_id[key_2]

                    pair_id = image_ids_to_pair_id(id_1, id_2)
                    if pair_id in added:
                        warnings.warn(f"Pair {pair_id} ({id_1}, {id_2}) already added!")
                        continue

                    matches = group[key_2][()]
                    db.add_matches(id_1, id_2, matches)

                    added.add(pair_id)

        return added

    def _create_camera(
        self,
        db: COLMAPDatabase,
        image_path: Path,
        camera_model: str,
    ) -> int:
        image = Image.open(image_path)
        width, height = image.size

        focal = self._get_focal(image_path)

        if camera_model == "simple-pinhole":
            model = 0  # simple pinhole
            param_arr = np.array([focal, width / 2, height / 2])
        if camera_model == "pinhole":
            model = 1  # pinhole
            param_arr = np.array([focal, focal, width / 2, height / 2])
        elif camera_model == "simple-radial":
            model = 2  # simple radial
            param_arr = np.array([focal, width / 2, height / 2, 0.1])
        elif camera_model == "opencv":
            model = 4  # opencv
            param_arr = np.array(
                [focal, focal, width / 2, height / 2, 0.0, 0.0, 0.0, 0.0]
            )

        return db.add_camera(model, width, height, param_arr)

    def _get_focal(self, image_path: Path, err_on_default=False):
        image = Image.open(image_path)
        max_size = max(image.size)

        exif = image.getexif()
        focal = None
        if exif is not None:
            focal_35mm = None
            # https://github.com/colmap/colmap/blob/d3a29e203ab69e91eda938d6e56e1c7339d62a99/src/util/bitmap.cc#L299
            for tag, value in exif.items():
                focal_35mm = None
                if ExifTags.TAGS.get(tag, None) == "FocalLengthIn35mmFilm":
                    focal_35mm = float(value)
                    break

            if focal_35mm is not None:
                focal = focal_35mm / 35.0 * max_size

        if focal is None:
            if err_on_default:
                raise RuntimeError("Failed to find focal length")

            # failed to find it in exif, use prior
            FOCAL_PRIOR = 1.2
            focal = FOCAL_PRIOR * max_size

        return focal
