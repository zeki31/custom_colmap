import multiprocessing as mp
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
import wandb
from tqdm import tqdm

from src.matching.tracking.trajectory import IncrementalTrajectorySet
from src.submodules.cotracker.predictor import CoTrackerPredictor

mp.set_start_method("spawn", force=True)


@dataclass
class TrackerCfg:
    ckpt_path: str = "checkpoints/scaled_offline.pth"
    window_len: int = 60
    grid_size: int = 60
    sample_ratio_grid: int = 6
    sample_ratio_aliked: int = 1
    traj_min_len: int = 2
    overlap: int = 2
    query: Literal["grid", "aliked"] = "grid"
    num_features: int = 4096
    viz: bool = False


class Tracker:
    def __init__(
        self,
        cfg: TrackerCfg,
        logger: wandb.sdk.wandb_run.Run,
        save_dir: Path,
    ):
        self.cfg = cfg
        self.logger = logger
        self.save_dir = save_dir

    def track(self, image_paths: list[Path], feature_dir: Path) -> None:
        """Execute the tracking process in parallel."""
        futures = []
        with ProcessPoolExecutor(max_workers=2) as executor:
            for i_proc, cam_name in enumerate(["2_dynA", "3_dynB", "4_dynC"]):
                sub_feature_dir = feature_dir / cam_name
                if (sub_feature_dir / "full_trajs_aliked.npy").exists():
                    print(f"Skipping {cam_name} as it already exists.")
                    continue

                sub_feature_dir.mkdir(parents=True, exist_ok=True)
                sub_image_paths = [
                    image_path
                    for image_path in image_paths
                    if cam_name in str(image_path)
                ]

                future = executor.submit(
                    self.track_point, sub_image_paths, sub_feature_dir, i_proc + 1
                )
                futures.append(future)
                print(f"Chunk {i_proc + 1}/3 submitted.")

            _ = [f.result() for f in futures]

    def track_point(
        self, image_paths: list[Path], feature_dir: Path, i_proc: int
    ) -> None:
        """Track point trajectories in the given frames."""
        if self.cfg.viz:
            self.viz_dir = feature_dir / "tracking_viz"
            self.viz_dir.mkdir(parents=True, exist_ok=True)

        gpu_id = 0 if i_proc % 2 else 1
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

        h, w = cv2.imread(image_paths[0]).shape[:2]
        trajs = IncrementalTrajectorySet(
            len(image_paths) + 1,
            h,
            w,
            self.cfg.sample_ratio_grid,
            self.cfg.sample_ratio_aliked,
            device,
            image_paths[0],
            self.cfg.query,
            self.cfg.num_features,
            init_frame_id=i_proc * len(image_paths),
        )

        point_tracker = CoTrackerPredictor(
            checkpoint=self.cfg.ckpt_path,
            v2=False,
            offline=True,
            window_len=60,
        ).to(device)

        start_t = 0
        stride = self.cfg.window_len - self.cfg.overlap
        n_aliked_queries = (
            trajs.candidate_desc.shape[0] if "aliked" in self.cfg.query else 0
        )
        with tqdm(total=len(image_paths) // stride + 1) as pbar:
            while start_t < len(image_paths):
                end_t = start_t + self.cfg.window_len

                frames = []
                for image_path in image_paths[start_t:end_t]:
                    im = cv2.imread(image_path)
                    frames.append(np.array(im))
                video = np.stack(frames)
                video = (
                    torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().to(device)
                )

                grid_pts = (
                    torch.from_numpy(trajs.sample_candidates)
                    .reshape(1, -1, 2)
                    .float()
                    .to(device)
                )
                queries = torch.cat(
                    [torch.ones_like(grid_pts[:, :, :1]) * 0, grid_pts],
                    dim=2,
                ).repeat(1, 1, 1)
                pred_tracks, pred_visibility = point_tracker(
                    video,
                    queries=queries,
                    grid_query_frame=0,
                    backward_tracking=True,
                )

                if self.cfg.viz and i_proc == 1:
                    from src.submodules.cotracker.utils.visualizer import Visualizer

                    vis = Visualizer(
                        save_dir=f"{self.viz_dir}",
                        pad_value=120,
                        linewidth=1,
                        fps=12,
                    )
                    vis.visualize(
                        video,
                        pred_tracks,
                        pred_visibility,
                        query_frame=0,
                        filename=f"{start_t:04d}",
                    )

                pred_tracks = pred_tracks[0].cpu().numpy()
                pred_visibility = pred_visibility[0].cpu().numpy()

                valid_cond = (
                    (pred_tracks[0][:, 0] >= 0)
                    & (pred_tracks[0][:, 0] < w)
                    & (pred_tracks[0][:, 1] >= 0)
                    & (pred_tracks[0][:, 1] < h)
                )
                viz_mask = (pred_visibility[0] > 0) & valid_cond
                loop = (
                    range(stride)
                    if len(pred_tracks) == self.cfg.window_len
                    else range(len(pred_tracks) - 1)
                )
                for timestep in loop:
                    frame_id = start_t + i_proc * len(image_paths) + timestep
                    # print(start_t, end_t, timestep, frame_id)

                    valid_cond = (
                        (pred_tracks[timestep][:, 0] >= 0)
                        & (pred_tracks[timestep][:, 0] < w)
                        & (pred_tracks[timestep][:, 1] >= 0)
                        & (pred_tracks[timestep][:, 1] < h)
                    )
                    viz_mask = viz_mask & (pred_visibility[timestep] > 0) & valid_cond
                    valid_cond_next = (
                        (pred_tracks[timestep + 1][viz_mask][:, 0] >= 0)
                        & (pred_tracks[timestep + 1][viz_mask][:, 0] < w)
                        & (pred_tracks[timestep + 1][viz_mask][:, 1] >= 0)
                        & (pred_tracks[timestep + 1][viz_mask][:, 1] < h)
                    )

                    # Propagate all the trajectories
                    if (
                        len(pred_tracks) == self.cfg.window_len
                        and timestep == stride - 1
                    ):
                        # Last timestep, we extend the active trajectories
                        n_aliked_queries = trajs.extend_all(
                            next_xys=pred_tracks[timestep + 1][viz_mask],
                            next_time=frame_id + 1,
                            flags=pred_visibility[timestep + 1][viz_mask]
                            & valid_cond_next,
                            next_descs=trajs.candidate_desc[viz_mask[:n_aliked_queries]]
                            if "aliked" in self.cfg.query
                            else None,
                            frame_path=image_paths[start_t + timestep + 1]
                            if start_t + timestep + 1 <= len(image_paths)
                            else image_paths[-1],
                        )
                    else:
                        trajs.extend_all(
                            next_xys=pred_tracks[timestep + 1][viz_mask],
                            next_time=frame_id + 1,
                            flags=pred_visibility[timestep + 1][viz_mask]
                            & valid_cond_next,
                            next_descs=trajs.candidate_desc[viz_mask[:n_aliked_queries]]
                            if "aliked" in self.cfg.query
                            else None,
                        )

                start_t += (
                    stride
                    if len(pred_tracks) == self.cfg.window_len
                    else len(pred_tracks)
                )

                pbar.update(1)

            trajs.clear_active()

        np.save(feature_dir / "full_trajs_grid.npy", trajs.full_trajs_grid)
        np.save(feature_dir / "full_trajs_aliked.npy", trajs.full_trajs_aliked)

        if self.cfg.viz and i_proc == 1:
            mp4_files = sorted(self.viz_dir.glob("*.mp4"))
            concat_list_path = self.viz_dir / "concat_list.txt"
            with open(concat_list_path, "w") as f:
                for mp4 in mp4_files:
                    f.write(f"file '{mp4.name}'\n")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "concat",
                    "-i",
                    str(concat_list_path),
                    "-c",
                    "copy",
                    str(self.viz_dir / "tracking.mp4"),
                ]
            )
            # self.logger.log(
            #     {
            #         "Tracking": wandb.Video(
            #             str(self.viz_dir / "tracking.mp4"), format="mp4"
            #         )
            #     }
            # )
