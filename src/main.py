import argparse
import gc
import shutil
import warnings
from pathlib import Path

import pycolmap
import torch
from jaxtyping import install_import_hook

from libs.pair_generator import get_pairs_exhaustive

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from libs.h5_to_db import import_into_colmap
    from libs.keypoint_detector import detect_keypoints
    from libs.keypoint_matcher import keypoint_distances
    from root_config import load_typed_root_config

warnings.simplefilter("ignore")


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description="""
        Run structure-from-motion using COLMAP.
        """
    )
    parser.add_argument(
        "--config_path",
        type=Path,
    )
    return parser.parse_args()


def main():
    args = get_arguments()
    cfg = load_typed_root_config(args.config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = cfg.base_dir / "images"
    image_paths = list(images_dir.glob(f"*.{cfg.ext}"))
    print(f"Got {len(image_paths)} images")

    feature_dir = cfg.output_dir / "feature_output"
    feature_dir.mkdir(parents=True, exist_ok=True)
    database_path = feature_dir / "colmap.db"
    if database_path.exists():
        database_path.unlink()

    # 1. Get the pairs of images
    index_pairs = get_pairs_exhaustive(
        image_paths,
    )
    gc.collect()

    # 2. Detect keypoints of all images
    detect_keypoints(
        image_paths,
        feature_dir,
        cfg.keypoint_detector,
        device=device,
    )
    gc.collect()

    # 3. Match  keypoints of pairs of similar images
    keypoint_distances(
        image_paths,
        index_pairs,
        feature_dir,
        cfg.keypoint_matcher,
        device=device,
    )
    gc.collect()

    # 4.1. Import keypoint distances of matches into colmap for RANSAC
    import_into_colmap(
        images_dir,
        feature_dir,
        database_path,
    )

    # 4.2. Compute RANSAC (detect match outliers)
    # By doing it exhaustively we guarantee we will find the best possible configuration
    pycolmap.match_exhaustive(database_path)

    # 5.1 Incrementally start reconstructing the scene (sparse reconstruction)
    # The process starts from a random pair of images and is incrementally extended by
    # registering new images and triangulating new points.
    if cfg.prior_dir is not None:
        (cfg.output_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)
        options = pycolmap.IncrementalPipelineOptions(
            ba_global_function_tolerance=0.000001,
            triangulation=pycolmap.IncrementalTriangulatorOptions(
                ignore_two_view_tracks=False, min_angle=0.1
            ),
        )
        reconstruction = pycolmap.Reconstruction(cfg.prior_dir)
        pycolmap.triangulate_points(
            reconstruction=reconstruction,
            database_path=database_path,
            image_path=images_dir,
            output_path=cfg.output_dir / "sparse" / "0",
            options=options,
        )
        (cfg.output_dir / "sparse" / "txt").mkdir(parents=True, exist_ok=True)
        reconstruction.write_text(cfg.output_dir / "sparse" / "txt")
    else:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        mapper_options = pycolmap.IncrementalPipelineOptions(
            max_num_models=cfg.colmap_mapper.max_num_models,
            min_model_size=cfg.colmap_mapper.min_model_size,
        )
        maps = pycolmap.incremental_mapping(
            database_path=database_path,
            image_path=images_dir,
            output_path=cfg.output_dir / "sparse",
            options=mapper_options,
        )

        # 5.2. Look for the best reconstruction: The incremental mapping offered by
        # pycolmap attempts to reconstruct multiple models, we must pick the best one
        images_registered = 0
        best_idx = None

        print("Looking for the best reconstruction")

        if isinstance(maps, dict):
            for idx1, rec in maps.items():
                print(idx1, rec.summary())
                try:
                    if len(rec.images) > images_registered:
                        images_registered = len(rec.images)
                        best_idx = idx1
                except Exception:
                    continue

        if isinstance(maps, dict):
            for idx, rec in maps.items():
                if idx == best_idx:
                    continue
                shutil.rmtree(cfg.output_dir / "sparse" / str(idx))

    # shutil.rmtree(feature_dir)


if __name__ == "__main__":
    main()
