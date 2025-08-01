from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import wandb

from src.colmap.read_write_model import read_model
from src.visualization.visualizer import Visualizer


@dataclass
class VisualizerRerunCfg:
    name: Literal["rerun"]
    filter: bool
    resize: int


class VisualizerRerun(Visualizer[VisualizerRerunCfg]):
    def __init__(
        self, cfg: VisualizerRerunCfg, logger: wandb.sdk.wandb_run.Run, save_dir: Path
    ):
        self.cfg = cfg
        self.logger = logger
        self.save_dir = save_dir

        self.filter_min_visible = 50
        self.description = """
            # Sparse Reconstruction by COLMAP
            This example was generated from the output of a sparse reconstruction done with COLMAP.

            [COLMAP](https://colmap.github.io/index.html) is a general-purpose Structure-from-Motion (SfM) and Multi-View Stereo
            (MVS) pipeline with a graphical and command-line interface.

            In this example a short video clip has been processed offline by the COLMAP pipeline, and we use Rerun to visualize the
            individual camera frames, estimated camera poses, and resulting point clouds over time.

            ## How it was made
            The full source code for this example is available
            [on GitHub](https://github.com/rerun-io/rerun/blob/latest/examples/python/structure_from_motion/main.py).

            ### Images
            The images are logged through the [rr.Image archetype](https://www.rerun.io/docs/reference/types/archetypes/image)
            to the [camera/image entity](recording://camera/image).

            ### Cameras
            The images stem from pinhole cameras located in the 3D world. To visualize the images in 3D, the pinhole projection has
            to be logged and the camera pose (this is often referred to as the intrinsics and extrinsics of the camera,
            respectively).

            The [rr.Pinhole archetype](https://www.rerun.io/docs/reference/types/archetypes/pinhole) is logged to
            the [camera/image entity](recording://camera/image) and defines the intrinsics of the camera. This defines how to go
            from the 3D camera frame to the 2D image plane. The extrinsics are logged as an
            [rr.Transform3D archetype](https://www.rerun.io/docs/reference/types/archetypes/transform3d) to the
            [camera entity](recording://camera).

            ### Reprojection error
            For each image a [rr.Scalar archetype](https://www.rerun.io/docs/reference/types/archetypes/scalar)
            containing the average reprojection error of the keypoints is logged to the
            [plot/avg_reproj_err entity](recording://plot/avg_reproj_err).

            ### 2D points
            The 2D image points that are used to triangulate the 3D points are visualized by logging
            [rr.Points3D archetype](https://www.rerun.io/docs/reference/types/archetypes/points2d)
            to the [camera/image/keypoints entity](recording://camera/image/keypoints). Note that these keypoints are a child of the
            [camera/image entity](recording://camera/image), since the points should show in the image plane.

            ### Colored 3D points
            The colored 3D points were added to the scene by logging the
            [rr.Points3D archetype](https://www.rerun.io/docs/reference/types/archetypes/points3d)
            to the [points entity](recording://points):
            ```python
            rr.log("points", rr.Points3D(points, colors=point_colors), rr.AnyValues(error=point_errors))
            ```
            **Note:** we added some [custom per-point errors](recording://points) that you can see when you
            hover over the points in the 3D view.
            """.strip()

    def read_and_log_sparse_reconstruction(
        self, base_dir: Path, save_dir: Path, filter_output: bool
    ) -> None:
        print("Reading sparse COLMAP reconstruction")
        cameras, images, points3D = read_model(save_dir / "sparse" / "0", ext=".bin")
        print("Building visualization by logging to Rerun")

        if filter_output:
            # Filter out noisy points
            points3D = {
                id: point
                for id, point in points3D.items()
                if point.rgb.any() and len(point.image_ids) > 2
            }

        rr.log(
            "description",
            rr.TextDocument(self.description, media_type=rr.MediaType.MARKDOWN),
            static=True,
        )
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        rr.log("plot/avg_reproj_err", rr.SeriesLines(colors=[240, 45, 58]), static=True)

        rr.set_time("frame", sequence=0)
        # Iterate through images (video frames) logging data related to each frame.
        for i, image in enumerate(sorted(images.values(), key=lambda im: im.name)):
            image_file = base_dir / image.name

            if not image_file.exists():
                print("skipping image", image_file, "because it does not exist")
                continue

            # COLMAP sets image ids that don't match the original video frame
            # idx_match = re.search(r"\d+", image.name)
            # assert idx_match is not None
            # frame_idx = int(idx_match.group(0))
            # print(idx_match, "and", frame_idx, "for image", image.name)
            # frame_idx = i  # Use the index in the sorted list as the frame index

            quat_xyzw = image.qvec[[1, 2, 3, 0]]  # COLMAP uses wxyz quaternions
            camera = cameras[image.camera_id]

            visible = [
                id != -1 and points3D.get(id) is not None for id in image.point3D_ids
            ]
            visible_ids = image.point3D_ids[visible]

            if filter_output and len(visible_ids) < self.filter_min_visible:
                continue

            visible_xyzs = [points3D[id] for id in visible_ids]
            visible_xys = image.xys[visible]

            points = [point.xyz for point in visible_xyzs]
            point_colors = [point.rgb for point in visible_xyzs]
            point_errors = [point.error for point in visible_xyzs]

            rr.log("plot/avg_reproj_err", rr.Scalars(np.mean(point_errors)))

            rr.log(
                f"points_{i:04d}",
                rr.Points3D(points, colors=point_colors),
                rr.AnyValues(error=point_errors),
            )

            # COLMAP's camera transform is "camera from world"
            rr.log(
                f"camera_{i:04d}",
                rr.Transform3D(
                    translation=image.tvec,
                    rotation=rr.Quaternion(xyzw=quat_xyzw),
                    relation=rr.TransformRelation.ChildFromParent,
                ),
            )
            rr.log(
                f"camera_{i:04d}", rr.ViewCoordinates.RDF, static=True
            )  # X=Right, Y=Down, Z=Forward

            # Log camera intrinsics
            if camera.model == "SIMPLE_PINHOLE":
                rr.log(
                    f"camera_{i:04d}/image",
                    rr.Pinhole(
                        resolution=[camera.width, camera.height],
                        focal_length=[camera.params[0], camera.params[0]],
                        principal_point=camera.params[1:],
                    ),
                )
            elif camera.model == "PINHOLE":
                rr.log(
                    f"camera_{i:04d}/image",
                    rr.Pinhole(
                        resolution=[camera.width, camera.height],
                        focal_length=camera.params[:2],
                        principal_point=camera.params[2:],
                    ),
                )
            else:
                raise ValueError(f"Unsupported camera model: {camera.model}")

            rr.log(f"camera_{i:04d}/image", rr.EncodedImage(path=base_dir / image.name))

            rr.log(
                f"camera_{i:04d}/image/keypoints",
                rr.Points2D(visible_xys, colors=[34, 138, 167]),
            )

    def viz(self, base_dir: Path, args: Namespace) -> None:
        """Visualize the sparse reconstruction."""
        blueprint = rrb.Vertical(
            rrb.Spatial3DView(
                name="3D",
                origin="/",
                line_grid=False,  # There's no clearly defined ground plane.
            ),
            rrb.Horizontal(
                rrb.TextDocumentView(name="README", origin="/description"),
                rrb.Spatial2DView(name="Camera", origin="/camera/image"),
                rrb.TimeSeriesView(origin="/plot"),
            ),
            row_shares=[3, 2],
        )

        rr.script_setup(
            args, "rerun_example_structure_from_motion", default_blueprint=blueprint
        )
        self.read_and_log_sparse_reconstruction(
            base_dir,
            save_dir=self.save_dir,
            filter_output=self.cfg.filter,
            # resize=self.cfg.resize,
        )
        rr.script_teardown(args)
