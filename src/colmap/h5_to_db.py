import os
import warnings
from pathlib import Path

import h5py
import numpy as np
from PIL import ExifTags, Image
from tqdm import tqdm

from src.colmap.database import COLMAPDatabase, image_ids_to_pair_id


def get_focal(image_path, err_on_default=False):
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


def create_camera(
    db: COLMAPDatabase,
    image_path: Path,
    camera_model: str,
) -> int:
    image = Image.open(image_path)
    width, height = image.size

    focal = get_focal(image_path)

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
        param_arr = np.array([focal, focal, width / 2, height / 2, 0.0, 0.0, 0.0, 0.0])

    return db.add_camera(model, width, height, param_arr)


def add_keypoints(
    db: COLMAPDatabase,
    h5_path: Path,
    image_path: Path,
    camera_model: str,
):
    keypoint_f = h5py.File(os.path.join(h5_path, "keypoints.h5"), "r")

    camera_id = None
    fname_to_id = {}
    for key in tqdm(list(keypoint_f.keys()), desc="Adding keypoints"):
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


def add_matches(
    db: COLMAPDatabase,
    h5_path: Path,
    fname_to_id: dict[str, int],
    added: set[int],
) -> set[int]:
    match_file = h5py.File(h5_path, "r")

    n_keys = len(match_file.keys())
    n_total = (n_keys * (n_keys - 1)) // 2

    with tqdm(total=n_total, desc="Adding matches") as pbar:
        for key_1 in match_file.keys():
            group = match_file[key_1]
            for key_2 in group.keys():
                id_1 = fname_to_id[key_1]
                id_2 = fname_to_id[key_2]

                pair_id = image_ids_to_pair_id(id_1, id_2)
                if pair_id in added:
                    warnings.warn(f"Pair {pair_id} ({id_1}, {id_2}) already added!")
                    continue

                matches = group[key_2][()]
                db.add_matches(id_1, id_2, matches)

                added.add(pair_id)

                pbar.update(1)

    return added


def add_fixed_kpts_matches(
    db: COLMAPDatabase,
    h5_path: Path,
    fname_to_id: dict[str, int],
    image_path: Path,
    camera_model: str,
) -> set[int]:
    match_file = h5py.File(h5_path, "r")

    camera_id = create_camera(db, image_path, camera_model)
    id_1 = db.add_image(image_path, camera_id)

    n_keys = len(match_file.keys())
    n_total = (n_keys * (n_keys - 1)) // 2

    added = set()
    with tqdm(total=n_total) as pbar:
        for key_1 in match_file.keys():
            group = match_file[key_1]
            for key_2 in group.keys():
                id_2 = fname_to_id[key_2]

                pair_id = image_ids_to_pair_id(id_1, id_2)
                if pair_id in added:
                    warnings.warn(f"Pair {pair_id} ({id_1}, {id_2}) already added!")
                    continue

                matches = group[key_2][()]
                db.add_matches(id_1, id_2, matches)

                added.add(pair_id)

                pbar.update(1)
