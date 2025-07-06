import argparse
import gc
import multiprocessing
import subprocess
import time
import warnings
from pathlib import Path

import pycolmap
import torch
import wandb
from jaxtyping import install_import_hook

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.colmap.importer import COLMAPImporter
    from src.matching import get_matcher
    from src.matching.retriever import Retriever
    from src.root_config import load_typed_root_config

warnings.simplefilter("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--cfg_path", type=Path)
    parser.add_argument("-o", "--overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_typed_root_config(args.cfg_path, args.overrides)

    save_dir = cfg.out_dir / cfg.wandb.project / cfg.wandb.name
    save_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(save_dir / "config.yaml")

    feature_dir = save_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    database_path = feature_dir / "colmap.db"
    if database_path.exists():
        database_path.unlink()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        dir=save_dir,
        config=cfg.to_dict(),
        mode=cfg.wandb.mode,
    ) as logger:
        retriever = Retriever(cfg.retriever, logger)
        image_paths = retriever.get_image_paths(cfg.base_dir)

        matcher = get_matcher(cfg.matcher, logger, device)
        matcher.match(image_paths, feature_dir)
        gc.collect()

        # Import keypoint distances of matches into colmap for RANSAC
        importer = COLMAPImporter(logger)
        importer.import_into_colmap(
            database_path, feature_dir, image_paths, cfg.matcher.name, cfg.base_dir
        )

        # Compute RANSAC (detect match outliers)
        # By doing it exhaustively we guarantee we will find the best possible configuration
        pycolmap.match_exhaustive(database_path)

        # Incrementally start reconstructing the scene (sparse reconstruction)
        # The process starts from a random pair of images and is incrementally extended by
        # registering new images and triangulating new points.
        if cfg.prior_dir is not None:
            (save_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)
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
                image_path=cfg.base_dir,
                output_path=save_dir / "sparse" / "0",
                options=options,
            )
            (save_dir / "sparse" / "txt").mkdir(parents=True, exist_ok=True)
            reconstruction.write_text(save_dir / "sparse" / "txt")
        else:
            save_dir.mkdir(parents=True, exist_ok=True)
            start = time.time()
            if cfg.mapper.name == "colmap":
                mapper_options = pycolmap.IncrementalPipelineOptions(
                    max_num_models=cfg.mapper.max_num_models,
                    min_model_size=cfg.mapper.min_model_size,
                    num_threads=min(multiprocessing.cpu_count(), 64),
                )
                pycolmap.incremental_mapping(
                    database_path=database_path,
                    image_path=cfg.base_dir,
                    output_path=save_dir / "sparse",
                    options=mapper_options,
                )

            elif cfg.mapper.name == "glomap":
                cmd = [
                    "glomap",
                    "mapper",
                    "--database_path",
                    str(database_path),
                    "--image_path",
                    str(cfg.base_dir),
                    "--output_path",
                    str((save_dir / "sparse")),
                    "--BundleAdjustment.use_gpu",
                    "1",
                    "--GlobalPositioning.use_gpu",
                    "1",
                ]
                subprocess.run(cmd, check=True)

            end = time.time()
            # Write out the time taken for mapping
            with open(save_dir / "mapping_time.txt", "w") as f:
                f.write(f"Mapping took {end - start:.2f} seconds\n")
                f.write(f"Mapping took {(end - start) / 60:.2f} minutes\n")

            logger.log({"mapping_time": (end - start) // 60})

        # shutil.rmtree(feature_dir)


if __name__ == "__main__":
    main()
