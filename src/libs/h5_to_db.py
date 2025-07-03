import os
import warnings
from pathlib import Path

import h5py
import numpy as np
from PIL import ExifTags, Image
from tqdm import tqdm

from .database import COLMAPDatabase, image_ids_to_pair_id
from .traj2matches import traj_to_matches


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
):
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
    for key in tqdm(list(keypoint_f.keys())):
        keypoints = keypoint_f[key][()]

        filename = key.replace("-", "/")
        path = os.path.join(image_path, filename)
        if not os.path.isfile(path):
            raise IOError(f"Invalid image path {path}")

        if camera_id is None:
            camera_id = create_camera(db, path, camera_model)
        image_id = db.add_image(filename, camera_id)
        fname_to_id[key] = image_id

        db.add_keypoints(image_id, keypoints)

    return fname_to_id


def add_matches(db, h5_path, fname_to_id):
    match_file = h5py.File(os.path.join(h5_path, "matches.h5"), "r")

    added = set()
    n_keys = len(match_file.keys())
    n_total = (n_keys * (n_keys - 1)) // 2

    with tqdm(total=n_total) as pbar:
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


def import_into_colmap(
    path: Path = None,
    feature_dir: Path = None,
    database_path: str = "colmap.db",
    image_paths: list[Path] = None,
) -> None:
    """Adds keypoints into colmap"""
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    # fname_to_id = add_keypoints(db, feature_dir, path, "simple-pinhole")
    # add_matches(
    #     db,
    #     feature_dir,
    #     fname_to_id,
    # )
    # db.commit()

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
