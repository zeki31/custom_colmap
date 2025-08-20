import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import pycolmap
import wandb


@dataclass
class MapperCfg:
    name: Literal["colmap", "glomap", "theia", "triangulation"] = "colmap"
    max_num_models: int = 2
    # By default colmap does not generate a reconstruction
    # if less than 10 images are registered. Lower it to 3.
    min_model_size: int = 3
    prior_dir: Optional[Path] = None


class Mapper:
    def __init__(self, cfg: MapperCfg, logger: wandb.sdk.wandb_run.Run, save_dir: Path):
        self.cfg = cfg
        self.logger = logger
        self.save_dir = save_dir

    def map(self, database_path: Path, base_dir: Path) -> None:
        # Compute RANSAC (detect match outliers)
        # By doing it exhaustively we guarantee we will find the best possible configuration
        # pycolmap.match_exhaustive(database_path)
        subprocess.run(
            [
                "colmap",
                "exhaustive_matcher",
                "--database_path",
                str(database_path),
                "--SiftMatching.gpu_index=0,1",
            ],
            check=True,
        )

        # Incrementally start reconstructing the scene (sparse reconstruction)
        # The process starts from a random pair of images and is incrementally extended by
        # registering new images and triangulating new points.
        start = time.time()
        output_dir = self.save_dir / "sparse"
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.name == "colmap":
            # mapper_options = pycolmap.IncrementalPipelineOptions(
            #     max_num_models=self.cfg.max_num_models,
            #     min_model_size=self.cfg.min_model_size,
            #     # num_threads=min(multiprocessing.cpu_count(), 64),
            # )
            # pycolmap.incremental_mapping(
            #     database_path=database_path,
            #     image_path=base_dir,
            #     output_path=save_dir / "sparse",
            #     options=mapper_options,
            # )
            cmd = [
                "colmap",
                "mapper",
                "--database_path",
                str(database_path),
                "--image_path",
                str(base_dir),
                "--output_path",
                str(output_dir),
                "--Mapper.max_num_models",
                str(self.cfg.max_num_models),
                "--Mapper.min_model_size",
                str(self.cfg.min_model_size),
                "--Mapper.ba_use_gpu",
                "1",
                "--Mapper.ba_gpu_index",
                "0,1",
                # "--log_level",
                # "2",
            ]
            subprocess.run(cmd)
        elif self.cfg.name == "glomap":
            cmd = [
                "glomap",
                "mapper",
                "--database_path",
                str(database_path),
                "--image_path",
                str(base_dir),
                "--output_path",
                str(output_dir),
                "--BundleAdjustment.use_gpu",
                "1",
                "--GlobalPositioning.use_gpu",
                "1",
            ]
            subprocess.run(cmd)
        elif self.cfg.name == "theia":
            curpath = Path(__file__).resolve().parents[1]
            theia_path = (
                curpath / "submodules" / "gmapper" / "build" / "src" / "exe" / "gcolmap"
            )
            cmd = [
                str(theia_path),
                "global_mapper",
                "--database_path",
                str(database_path),
                "--image_path",
                str(base_dir),
                "--output_path",
                str(output_dir),
            ]
            subprocess.run(cmd)
        elif self.cfg.name == "triangulation":
            assert (
                self.cfg.prior_dir is not None
            ), "Prior directory must be specified for triangulation."

            (output_dir / "0").mkdir(parents=True, exist_ok=True)
            options = pycolmap.IncrementalPipelineOptions(
                ba_global_function_tolerance=0.000001,
                triangulation=pycolmap.IncrementalTriangulatorOptions(
                    ignore_two_view_tracks=False, min_angle=0.1
                ),
            )
            reconstruction = pycolmap.Reconstruction(self.cfg.prior_dir)
            pycolmap.triangulate_points(
                reconstruction=reconstruction,
                database_path=database_path,
                image_path=base_dir,
                output_path=output_dir / "0",
                options=options,
            )

        end = time.time()

        print(f"Mapping took {end - start:.2f} seconds")
        print(f"Mapping took {(end - start) / 60:.2f} minutes")

        reconstruction = pycolmap.Reconstruction(self.save_dir / "sparse" / "0")
        print(f"# of registered images: {len(reconstruction.images)}")
        print(f"# of registered points: {len(reconstruction.points3D)}")
        print(
            f"Mean Reprojection Error (px): {reconstruction.compute_mean_reprojection_error():.2f}"
        )
        print(f"Mean Track Length: {reconstruction.compute_mean_track_length():.2f}")
        print(
            f"Mean Observations per Registered Image: {reconstruction.compute_mean_observations_per_reg_image():.2f}"
        )
        self.logger.log(
            {
                "Mapping time (min)": (end - start) // 60,
                "# of registered images": len(reconstruction.images),
                "# of registered points": len(reconstruction.points3D),
                "Mean reprojection error (px)": reconstruction.compute_mean_reprojection_error(),
                "Mean track length": reconstruction.compute_mean_track_length(),
                "Mean observations per registered image": reconstruction.compute_mean_observations_per_reg_image(),
            }
        )

    def register_imgs(self, database_path: Path, input_dir: Path) -> None:
        subprocess.run(
            [
                "colmap",
                "exhaustive_matcher",
                "--database_path",
                str(database_path),
                "--SiftMatching.gpu_index=0,1",
            ],
            check=True,
        )
        start = time.time()

        cmd = [
            "colmap",
            "image_registrator",
            "--database_path",
            str(database_path),
            "--input_path",
            str(input_dir),
            "--output_path",
            str(input_dir),
        ]
        subprocess.run(cmd)

        end = time.time()

        print(f"Registration took {end - start:.2f} seconds")
        print(f"Registration took {(end - start) / 60:.2f} minutes")

        reconstruction = pycolmap.Reconstruction(input_dir)
        print(f"# of registered images: {len(reconstruction.images)}")
        print(f"# of registered points: {len(reconstruction.points3D)}")
        print(
            f"Mean Reprojection Error (px): {reconstruction.compute_mean_reprojection_error():.2f}"
        )
        print(f"Mean Track Length: {reconstruction.compute_mean_track_length():.2f}")
        print(
            f"Mean Observations per Registered Image: {reconstruction.compute_mean_observations_per_reg_image():.2f}"
        )
        self.logger.log(
            {
                "Registration time (min)": (end - start) // 60,
                "# of registered images Aft.reg": len(reconstruction.images),
                "# of registered points Aft.reg": len(reconstruction.points3D),
                "Mean reprojection error (px) Aft.reg": reconstruction.compute_mean_reprojection_error(),
                "Mean track length Aft.reg": reconstruction.compute_mean_track_length(),
                "Mean observations per registered image Aft.reg": reconstruction.compute_mean_observations_per_reg_image(),
            }
        )

    def bundle_adjustment(self, input_dir: Path) -> None:
        start = time.time()

        cmd = [
            "colmap",
            "bundle_adjuster",
            "--input_path",
            str(input_dir),
            "--output_path",
            str(input_dir),
            "--BundleAdjustment.use_gpu=1",
        ]
        subprocess.run(cmd)

        end = time.time()

        print(f"Bundle adjustment took {end - start:.2f} seconds")
        print(f"Bundle adjustment took {(end - start) / 60:.2f} minutes")

        reconstruction = pycolmap.Reconstruction(input_dir)
        print(f"# of registered images: {len(reconstruction.images)}")
        print(f"# of registered points: {len(reconstruction.points3D)}")
        print(
            f"Mean Reprojection Error (px): {reconstruction.compute_mean_reprojection_error():.2f}"
        )
        print(f"Mean Track Length: {reconstruction.compute_mean_track_length():.2f}")
        print(
            f"Mean Observations per Registered Image: {reconstruction.compute_mean_observations_per_reg_image():.2f}"
        )
        self.logger.log(
            {
                "Bundle adjustment time (min)": (end - start) // 60,
                "# of registered images Aft.ba": len(reconstruction.images),
                "# of registered points Aft.ba": len(reconstruction.points3D),
                "Mean reprojection error (px) Aft.ba": reconstruction.compute_mean_reprojection_error(),
                "Mean track length Aft.ba": reconstruction.compute_mean_track_length(),
                "Mean observations per registered image Aft.ba": reconstruction.compute_mean_observations_per_reg_image(),
            }
        )
