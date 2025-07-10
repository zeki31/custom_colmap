import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

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
    sample_ratio: int = 6
    traj_min_len: int = 2
    overlap: int = 2


class Tracker:
    def __init__(
        self,
        cfg: TrackerCfg,
        logger: wandb.sdk.wandb_run.Run,
    ):
        self.cfg = cfg
        self.logger = logger

    def track(self, image_paths: list[Path], feature_dir: Path) -> None:
        """Execute the tracking process in parallel."""
        futures = []
        with ProcessPoolExecutor(max_workers=2) as executor:
            for i_proc, cam_name in enumerate(["2_dynA"]):
                sub_feature_dir = feature_dir / cam_name
                if (sub_feature_dir / "full_trajs.npy").exists():
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
        gpu_id = 0 if i_proc % 2 else 1
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

        h, w = cv2.imread(image_paths[0]).shape[:2]
        stride = self.cfg.window_len - self.cfg.overlap
        trajs = IncrementalTrajectorySet(
            len(image_paths) + 1, h, w, self.cfg.sample_ratio, device, image_paths[0]
        )

        point_tracker = CoTrackerPredictor(
            checkpoint=self.cfg.ckpt_path,
            v2=False,
            offline=True,
            window_len=60,
        ).to(device)

        start_t = 0
        with tqdm(total=len(image_paths) // stride + 1) as pbar:
            while start_t < len(image_paths):
                end_t = start_t + self.cfg.window_len

                frames = []
                for image_path in image_paths[start_t:end_t]:
                    im = cv2.imread(str(image_path))
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

                # # Save a video with predicted tracks
                # from src.submodules.cotracker.utils.visualizer import Visualizer
                # vis = Visualizer(
                #     save_dir=f"results/tracking_aliked_12fps10win_{i_proc}",
                #     pad_value=120,
                #     linewidth=1,
                #     fps=12,
                # )
                # vis.visualize(
                #     video,
                #     pred_tracks,
                #     pred_visibility,
                #     query_frame=0,
                #     filename=f"track_{start_t:04d}_{end_t:04d}",
                # )

                pred_tracks = pred_tracks[0].cpu().numpy()
                pred_visibility = pred_visibility[0].cpu().numpy()

                valid_cond = (
                    (pred_tracks[0][:, 0] > 0)
                    & (pred_tracks[0][:, 0] < w - 1)
                    & (pred_tracks[0][:, 1] > 0)
                    & (pred_tracks[0][:, 1] < h - 1)
                )
                viz_mask = (pred_visibility[0] > 0) & valid_cond
                for timestep in range(len(pred_tracks) - 1):
                    frame_id = start_t + i_proc * len(image_paths) + timestep

                    # Generate new trajectories if needed
                    if start_t == 0 and timestep == 0:
                        points = pred_tracks[timestep]
                        times = (np.ones(points.shape[0]) * frame_id).astype(int)
                        trajs.new_traj_all(times, points)

                    valid_cond = (
                        (pred_tracks[timestep][:, 0] > 0)
                        & (pred_tracks[timestep][:, 0] < w - 1)
                        & (pred_tracks[timestep][:, 1] > 0)
                        & (pred_tracks[timestep][:, 1] < h - 1)
                    )
                    viz_mask = viz_mask & (pred_visibility[timestep] > 0) & valid_cond
                    valid_cond_next = (
                        (pred_tracks[timestep + 1][viz_mask][:, 0] > 0)
                        & (pred_tracks[timestep + 1][viz_mask][:, 0] < w - 1)
                        & (pred_tracks[timestep + 1][viz_mask][:, 1] > 0)
                        & (pred_tracks[timestep + 1][viz_mask][:, 1] < h - 1)
                    )

                    # Propagate all the trajectories
                    if timestep == len(pred_tracks) - 2:
                        # Last timestep, we extend the active trajectories
                        trajs.extend_all(
                            pred_tracks[timestep + 1][viz_mask],
                            frame_id + 1,
                            pred_visibility[timestep + 1][viz_mask] & valid_cond_next,
                            trajs.candidate_desc[viz_mask],
                            image_paths[end_t - 1]
                            if end_t < len(image_paths)
                            else image_paths[-1],
                        )
                    else:
                        trajs.extend_all(
                            pred_tracks[timestep + 1][viz_mask],
                            frame_id + 1,
                            pred_visibility[timestep + 1][viz_mask] & valid_cond_next,
                            trajs.candidate_desc[viz_mask],
                        )

                start_t += stride

                pbar.update(1)

            trajs.clear_active()

        np.save(feature_dir / "full_trajs.npy", trajs.full_trajs)
