from pathlib import Path
from typing import Literal, Optional, Union

import kornia as K
import numpy as np
import scipy
import torch
from jaxtyping import Bool, Float, Int
from numpy.typing import NDArray
from tqdm import tqdm

from src.submodules.LightGlue.lightglue import ALIKED


class Trajectory(object):
    def __init__(
        self,
        start_time: int,
        start_xy: Float[NDArray, "2"],
        start_desc: Optional[Float[NDArray, " D"]] = None,
    ):
        self.times = []
        self.xys = []
        self.descs = []
        self.extend(start_time, start_xy, start_desc)

    def extend(
        self,
        time: int,
        xy: Float[NDArray, "2"],
        desc: Optional[Float[NDArray, " D"]] = None,
    ):
        self.times.append(time)
        self.xys.append(xy)
        if desc is not None:
            self.descs.append(desc)

    def length(self):
        return len(self.xys)

    def get_tail_location(self):
        if self.length() == 0:
            raise ValueError("Error!The trajectory is empty!")
        return self.xys[-1]

    def as_dict(self):
        return {
            "frame_ids": self.times,
            "locations": self.xys,
        }


class IncrementalTrajectorySet(object):
    def __init__(
        self,
        total_length: int,
        img_h: int,
        img_w: int,
        sample_ratio_grid: int,
        sample_ratio_aliked: int,
        device: torch.device,
        init_frame_path: Path,
        query: Literal["grid", "aliked"],
        num_features: int,
        init_frame_id: int,
    ):
        self.total_length = total_length
        self.ratio_grid = sample_ratio_grid
        self.ratio_aliked = sample_ratio_aliked
        self.h, self.w = img_h, img_w
        self.active_trajs_grid = []
        self.active_trajs_aliked = []
        self.full_trajs_grid = []
        self.full_trajs_aliked = []
        self.query = query
        self.num_features = num_features

        self.device = device
        self.dtype = torch.float32  # ALIKED has issues with float16
        self.extractor = (
            ALIKED(
                max_num_keypoints=self.num_features,
                detection_threshold=0.01,
                # resize=1024,
            )
            .eval()
            .to(self.device, self.dtype)
        )

        # Compute the initial candidates (= queries)
        (
            candidate_kpts,
            self.candidate_desc,
        ) = self.generate_aliked_candidates(init_frame_path)
        self.grid_all_candidates = self.generate_grid_candidates()

        occupied_map = np.zeros((self.h, self.w, 1))
        occupied_map[
            candidate_kpts[:, 1].astype(int), candidate_kpts[:, 0].astype(int)
        ] = 1
        occupied_map_trans = scipy.ndimage.morphology.distance_transform_edt(
            1.0 - occupied_map
        )
        sample_map = (occupied_map_trans > self.ratio_grid)[
            :: self.ratio_grid, :: self.ratio_grid, 0
        ]
        candidate_grid = np.copy(self.grid_all_candidates[sample_map])
        self.n_max_grid = 3600 * 4
        print(
            "ALIKED candidates:",
            candidate_kpts.shape[0],
            "Grid candidates:",
            candidate_grid.shape[0],
        )
        # Combine the candidates
        self.sample_candidates = np.concatenate(
            [candidate_kpts, candidate_grid], axis=0
        )

        # Initialize the trajectories
        self.new_traj_all(
            start_times=(np.ones(candidate_kpts.shape[0]) * init_frame_id).astype(int),
            start_xys=candidate_kpts,
            start_desc=self.candidate_desc,
        )
        self.new_traj_all(
            start_times=(np.ones(candidate_grid.shape[0]) * init_frame_id).astype(int),
            start_xys=candidate_grid,
        )

    def _load_torch_image(self, file_name: Path | str, device=torch.device("cpu")):
        """Loads an image and adds batch dimension"""
        img = K.io.load_image(file_name, K.io.ImageLoadType.RGB32, device=device)[
            None, ...
        ]
        return img

    def generate_aliked_candidates(
        self, frame_path: Path
    ) -> tuple[Float[NDArray, "N 2"], Float[NDArray, "N D"]]:
        with torch.inference_mode():
            image = self._load_torch_image(frame_path, device=self.device).to(
                self.dtype
            )
            features = self.extractor.extract(image)
        kpts = features["keypoints"].detach().cpu().numpy().squeeze(0)  # shape: (N, 2)
        descs = features["descriptors"].detach().cpu().numpy().squeeze(0)
        # valid_cond = (
        #     (kpts[:, 0] > 0)
        #     & (kpts[:, 0] < self.w - 1)
        #     & (kpts[:, 1] > 0)
        #     & (kpts[:, 1] < self.h - 1)
        # )
        # kpts = kpts[valid_cond].copy()  # shape: (N_valid, 2)
        # descs = descs[valid_cond].copy()
        return kpts, descs

    def generate_grid_candidates(self) -> Float[NDArray, "h_sampled w_sampled 2"]:
        """Grid sampling of the image."""
        x, y = np.arange(0, self.w), np.arange(0, self.h)
        xx, yy = np.meshgrid(x, y)
        xys = np.stack([xx, yy], -1, dtype=np.float32)
        s_xys = xys[:: self.ratio_grid, :: self.ratio_grid, :]
        return s_xys

    def new_traj_all(
        self,
        start_times: Int[NDArray, " N"],
        start_xys: Float[NDArray, "N 2"],
        start_desc=None,
    ):
        """Append new Trajectory to the active trajectories.
        If start_desc is None, the trajectories are added to trajs_grid."""
        if start_desc is None:
            for time, xy in zip(start_times, start_xys):
                t = Trajectory(int(time), xy)
                self.active_trajs_grid.append(t)
        else:
            for time, xy, desc in zip(start_times, start_xys, start_desc):
                t = Trajectory(int(time), xy, desc)
                self.active_trajs_aliked.append(t)

    def get_cur_pos(
        self, return_desc: bool = False
    ) -> Union[
        Float[NDArray, "N_curr 2"],
        tuple[Float[NDArray, "N_curr 2"], Float[NDArray, "N_curr D"]],
    ]:
        # Get all the current traj positions
        cur_pos = []
        if not return_desc:
            for i in range(len(self.active_trajs_grid)):
                cur_pos.append(self.active_trajs_grid[i].get_tail_location())
            return np.array(cur_pos)

        cur_desc = []
        for i in range(len(self.active_trajs_aliked)):
            cur_pos.append(self.active_trajs_aliked[i].get_tail_location())
            cur_desc.append(self.active_trajs_aliked[i].descs[-1])
        return np.array(cur_pos), np.array(cur_desc)

    def extend_all(
        self,
        next_xys: Float[NDArray, "N 2"],
        next_time: int,
        flags: Bool[NDArray, " N"],
        next_descs: Optional[Float[NDArray, "N_aliked D"]] = None,
        frame_path: Path = None,
    ) -> int | None:
        n_aliked_queries = len(self.active_trajs_aliked)
        n_grid_queries = len(self.active_trajs_grid)
        # print(f"Extending {len(self.active_trajs_grid)} + {len(self.active_trajs_aliked)} = {len(self.active_trajs_grid) + len(self.active_trajs_aliked)} active trajectories with {len(next_xys)} new points at time {next_time}.")
        # Extend all the trajs
        assert len(next_xys) == (
            n_aliked_queries + n_grid_queries
        ), f"{len(next_xys)} != {n_aliked_queries} + {n_grid_queries}"
        assert len(flags) == len(next_xys)

        # To check the next sample candidates
        occupied_map = np.zeros((self.h, self.w, 1))
        new_active_trajs_aliked = []
        new_active_trajs_grid = []
        # Handle the trajectories from ALIKED
        for i in range(n_aliked_queries):
            next_xy, flag, next_desc = next_xys[i], flags[i], next_descs[i]
            if not flag:
                self.full_trajs_aliked.append(self.active_trajs_aliked[i])
            else:
                occupied_map[int(next_xy[1]), int(next_xy[0])] = 1
                self.active_trajs_aliked[i].extend(next_time, next_xy, next_desc)
                new_active_trajs_aliked.append(self.active_trajs_aliked[i])
        # Handle the trajectories from the grid sampling
        for i in range(n_grid_queries):
            next_xy, flag = (
                next_xys[i + n_aliked_queries],
                flags[i + n_aliked_queries],
            )
            if not flag:
                self.full_trajs_grid.append(self.active_trajs_grid[i])
            else:
                occupied_map[int(next_xy[1]), int(next_xy[0])] = 1
                self.active_trajs_grid[i].extend(next_time, next_xy)
                new_active_trajs_grid.append(self.active_trajs_grid[i])

        self.active_trajs_aliked = new_active_trajs_aliked
        self.active_trajs_grid = new_active_trajs_grid

        if frame_path is None:
            return

        occupied_map_trans = scipy.ndimage.morphology.distance_transform_edt(
            1.0 - occupied_map
        )  # [H, W, 1]

        # Generate the next sample candidates using ALIKED
        extracted_pts, extracted_descs = self.generate_aliked_candidates(
            frame_path
        )  # [N, 2]
        active_pts_aliked, active_pts_desc = self.get_cur_pos(return_desc=True)
        # Sample the candidates that are not occupied
        xs = extracted_pts[:, 0].astype(int)
        ys = extracted_pts[:, 1].astype(int)
        # If occupied_map_trans has shape (H, W, 1), squeeze the last dimension
        sample_map = occupied_map_trans[ys, xs].squeeze() > self.ratio_aliked  # (N,)
        non_active_candidates_aliked = extracted_pts[sample_map]
        non_active_candidates_desc = extracted_descs[sample_map]
        # Reduce the candidates if there are too many
        n_non_active_needed = max(0, self.num_features - len(active_pts_aliked))
        if len(non_active_candidates_aliked) > n_non_active_needed:
            idx = np.random.choice(
                len(non_active_candidates_aliked), n_non_active_needed, replace=False
            )
            non_active_candidates_aliked = non_active_candidates_aliked[idx]
            non_active_candidates_desc = non_active_candidates_desc[idx]

        times = (np.ones(non_active_candidates_aliked.shape[0]) * next_time).astype(int)
        self.new_traj_all(
            times, non_active_candidates_aliked, non_active_candidates_desc
        )
        self.candidate_desc = np.concatenate(
            [active_pts_desc, non_active_candidates_desc], axis=0
        )

        occupied_map[
            non_active_candidates_aliked[:, 1].astype(int),
            non_active_candidates_aliked[:, 0].astype(int),
        ] = 1
        occupied_map_trans = scipy.ndimage.morphology.distance_transform_edt(
            1.0 - occupied_map
        )

        # Generate the next sample candidates using grid sampling
        sample_map = (occupied_map_trans > self.ratio_grid)[
            :: self.ratio_grid, :: self.ratio_grid, 0
        ]
        active_pts_grid = self.get_cur_pos()
        non_active_candidates_grid = np.copy(self.grid_all_candidates[sample_map])

        # curr_total = (
        #     len(active_pts_grid)
        #     + len(non_active_candidates_aliked)
        #     + len(active_pts_grid)
        #     + len(non_active_candidates_grid)
        # )
        # if curr_total > self.n_max_grid:
        #     idx = np.random.choice(
        #         non_active_candidates_grid.shape[0],
        #         self.n_max_grid - curr_total,
        #         replace=False,
        #     )
        #     non_active_candidates_grid = non_active_candidates_grid[idx]

        self.new_traj_all(
            start_times=(
                np.ones(non_active_candidates_grid.shape[0]) * next_time
            ).astype(int),
            start_xys=non_active_candidates_grid,
        )

        self.sample_candidates = np.concatenate(
            [
                active_pts_aliked,
                non_active_candidates_aliked,
                active_pts_grid,
                non_active_candidates_grid,
            ],
            axis=0,
        )

        # print(
        #     active_pts_aliked.shape[0],
        #     non_active_candidates_aliked.shape[0],
        #     active_pts_grid.shape[0],
        #     non_active_candidates_grid.shape[0],
        # )

        return len(
            np.concatenate([active_pts_aliked, non_active_candidates_aliked], axis=0)
        )

    def clear_active(self):
        for traj in self.active_trajs_aliked:
            self.full_trajs_aliked.append(traj)
        for traj in self.active_trajs_grid:
            self.full_trajs_grid.append(traj)
        self.active_trajs_aliked = []
        self.active_trajs_grid = []


class TrajectorySet:
    """
    A Python implementation of the C++ TrajectorySet.
    Relies on a Python Trajectory class with attributes:
      - times: list of frame IDs
      - xys: list of (x, y) locations
      - as_dict(): returns a dict with keys 'frame_ids', 'locations', 'labels'
    """

    def __init__(self, trajs: dict[int, Trajectory]):
        self.trajs = trajs

    def as_dict(self):
        """
        Returns a dict mapping traj_id -> Trajectory.as_dict().
        """
        output = {}
        for traj_id, traj in self.trajs.items():
            output[traj_id] = traj.as_dict()
        return output

    def build_invert_indexes(self):
        """Build invert_maps: frame_id -> {traj_id -> index}"""
        self.invert_maps = {}
        for traj_id, traj in tqdm(self.trajs.items(), desc="Building invert indexes"):
            for i in range(traj.length()):
                frame_id = traj.times[i]
                if frame_id not in self.invert_maps:
                    self.invert_maps[frame_id] = {}
                self.invert_maps[frame_id][traj_id] = i
