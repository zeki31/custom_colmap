from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import wandb
from tqdm import tqdm

from src.matching.tracking.trajectory import IncrementalTrajectorySet, TrajectorySet
from src.submodules.cotracker.predictor import CoTrackerPredictor


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
        device: torch.device,
    ):
        self.cfg = cfg
        self.logger = logger
        self.device = device

        self.point_tracker = CoTrackerPredictor(
            checkpoint=self.cfg.ckpt_path,
            v2=False,
            offline=True,
            window_len=60,
        ).to(self.device)

    def track(self, image_paths: list[Path], feature_dir: Path) -> None:
        """Sequentially track point trajectories"""
        if (feature_dir / "track.npy").exists():
            return

        h, w = cv2.imread(image_paths[0]).shape[:2]
        stride = self.cfg.window_len - self.cfg.overlap
        trajs = IncrementalTrajectorySet(
            len(image_paths) + 1, h, w, self.cfg.sample_ratio
        )

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
                    torch.from_numpy(video)
                    .permute(0, 3, 1, 2)[None]
                    .float()
                    .to(self.device)
                )

                grid_pts = (
                    torch.from_numpy(trajs.sample_candidates)
                    .reshape(1, -1, 2)
                    .float()
                    .to(self.device)
                )
                queries = torch.cat(
                    [torch.ones_like(grid_pts[:, :, :1]) * 0, grid_pts],
                    dim=2,
                ).repeat(1, 1, 1)
                pred_tracks, pred_visibility = self.point_tracker(
                    video,
                    queries=queries,
                    grid_query_frame=0,
                    backward_tracking=True,
                )

                # # Save a video with predicted tracks
                # from .submodules.cotracker.utils.visualizer import Visualizer
                # vis = Visualizer(save_dir="results/tracking", pad_value=120, linewidth=1, fps=60)
                # vis.visualize(
                #     video,
                #     pred_tracks,
                #     pred_visibility,
                #     query_frame=0,
                #     filename=f"track_{start_t:04d}_{end_t:04d}",
                # )

                pred_tracks = pred_tracks[0].cpu().numpy()
                pred_visibility = pred_visibility[0].cpu().numpy()

                viz_mask = pred_visibility[0] > 0
                for timestep in range(len(pred_tracks) - 1):
                    frame_id = start_t + timestep

                    # Generate new trajectories if needed
                    if start_t == 0 and timestep == 0:
                        points = pred_tracks[timestep]
                        times = (np.ones(points.shape[0]) * frame_id).astype(int)
                        trajs.new_traj_all(times, points)

                    viz_mask = viz_mask & (pred_visibility[timestep] > 0)

                    # Propagate all the trajectories
                    if timestep == len(pred_tracks) - 2:
                        # Last timestep, we extend the active trajectories
                        trajs.extend_all(
                            pred_tracks[timestep + 1][viz_mask],
                            frame_id + 1,
                            pred_visibility[timestep + 1][viz_mask],
                            add_non_active=True,
                        )
                    else:
                        trajs.extend_all(
                            pred_tracks[timestep + 1][viz_mask],
                            frame_id + 1,
                            pred_visibility[timestep + 1][viz_mask],
                        )

                start_t += stride

                pbar.update(1)

            trajs.clear_active()

        # Save the outputs
        dict_trajs = {}
        for idx, traj in enumerate(trajs.full_trajs):
            if traj.length() < self.cfg.traj_min_len:
                continue
            dict_trajs[idx] = traj
        trajectories = TrajectorySet(dict_trajs)
        np.save(feature_dir / "track.npy", trajectories)
