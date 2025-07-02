from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .submodules.cotracker.predictor import CoTrackerPredictor
from .trajectory import IncrementalTrajectorySet, TrajectorySet


@dataclass
class TrackingCfg:
    ckpt_path: str = "checkpoints/scaled_offline.pth"
    window_len: int = 60
    grid_size: int = 60
    sample_ratio: int = 6
    traj_min_len: int = 3


def track(
    cfg: TrackingCfg, image_paths: list[Path], output_dir: Path, device: torch.device
):
    """Sequentially track point trajectories"""
    if (output_dir / "track.npy").exists():
        return

    cotracker3 = CoTrackerPredictor(
        checkpoint=cfg.ckpt_path,
        v2=False,
        offline=True,
        window_len=60,
    )
    cotracker3 = cotracker3.to(device)

    h, w = 270, 480
    trajs = IncrementalTrajectorySet(len(image_paths) + 1, h, w, cfg.sample_ratio)
    for i in tqdm(range(len(image_paths) // cfg.window_len), desc="Tracking"):
        start_t = i * cfg.window_len
        end_t = start_t + cfg.window_len

        frames = []
        for image_path in image_paths[start_t:end_t]:
            im = cv2.imread(str(image_path))
            im_resized = cv2.resize(im, (480, 270))
            frames.append(np.array(im_resized))
        video = np.stack(frames)
        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().to(device)

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
        pred_tracks, pred_visibility = cotracker3(
            video,
            queries=queries,
            grid_query_frame=0,
            backward_tracking=True,
        )

        pred_tracks = pred_tracks[0].cpu().numpy()
        pred_visibility = pred_visibility[0].cpu().numpy()

        viz_mask = pred_visibility[0] > 0
        for timestep in range(cfg.window_len - 1):
            frame_id = start_t + timestep

            # Generate new trajectories if needed
            if timestep == 0:
                points = trajs.sample_candidates
                times = (np.ones(points.shape[0]) * frame_id).astype(int)
                trajs.new_traj_all(times, points)

            viz_mask = viz_mask & (pred_visibility[timestep] > 0)

            # Propagate all the trajectories
            trajs.extend_all(
                pred_tracks[timestep + 1][viz_mask],
                frame_id + 1,
                pred_visibility[timestep + 1][viz_mask],
            )

        trajs.clear_active()

    # save the outputs
    dict_trajs = {}
    for idx, traj in enumerate(trajs.full_trajs):
        if traj.length() < cfg.traj_min_len:
            continue
        dict_trajs[idx] = traj
    trajectories = TrajectorySet(dict_trajs)
    np.save(output_dir / "track.npy", trajectories)
