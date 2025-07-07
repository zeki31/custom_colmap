from pathlib import Path

import kornia as K
import numpy as np
import scipy
import torch
from jaxtyping import Float
from numpy.typing import NDArray
from tqdm import tqdm

from src.submodules.LightGlue.lightglue import ALIKED


class Trajectory(object):
    def __init__(self, start_time, start_xy, start_desc):
        self.times = []
        self.xys = []
        self.descs = []
        self.extend(start_time, start_xy, start_desc)

    def extend(self, time, xy, desc):
        self.times.append(time)
        self.xys.append(xy)
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
        sample_ratio: int,
        device: torch.device,
        init_frame_path: Path,
    ):
        self.total_length = total_length
        self.ratio = sample_ratio
        self.h, self.w = img_h, img_w
        self.active_trajs = []
        self.full_trajs = []

        self.device = device
        self.dtype = torch.float32  # ALIKED has issues with float16
        self.extractor = (
            ALIKED(
                max_num_keypoints=4096,
                detection_threshold=0.01,
                # resize=1024,
            )
            .eval()
            .to(self.device, self.dtype)
        )

        self.sample_candidates = self.generate_all_candidates(init_frame_path)

    def _load_torch_image(self, file_name: Path | str, device=torch.device("cpu")):
        """Loads an image and adds batch dimension"""
        img = K.io.load_image(file_name, K.io.ImageLoadType.RGB32, device=device)[
            None, ...
        ]
        return img

    def generate_all_candidates(self, frame_path: Path) -> Float[NDArray, "N 2"]:
        # # Grid sampling of the image
        # x, y = np.arange(0, self.w), np.arange(0, self.h)
        # xx, yy = np.meshgrid(x, y)
        # xys = np.stack([xx, yy], -1)
        # s_xys = xys[:: self.ratio, :: self.ratio, :]

        with torch.inference_mode():
            image = self._load_torch_image(frame_path, device=self.device).to(
                self.dtype
            )
            features = self.extractor.extract(image)
        kpts = features["keypoints"].detach().cpu().numpy()  # shape: (N, 2)
        return kpts

    def new_traj_all(self, start_times, start_xys):
        for time, xy in zip(start_times, start_xys):
            t = Trajectory(time, xy)
            self.active_trajs.append(t)

    def get_cur_pos(self):
        # Get all the current traj positions
        cur_pos = []
        for i in range(len(self.active_trajs)):
            cur_pos.append(self.active_trajs[i].get_tail_location())
        return np.array(cur_pos)

    def extend_all(self, next_xys, next_time: int, flags, frame_path: Path = None):
        # Extend all the trajs
        assert len(next_xys) == len(
            self.active_trajs
        ), f"{len(next_xys)} != {len(self.active_trajs)}"
        assert len(flags) == len(next_xys)

        # also check the next sample candidates
        occupied_map = np.zeros((self.h, self.w, 1))

        new_active_trajs = []
        for i in range(len(flags)):
            next_xy, flag = next_xys[i], flags[i]
            if not flag:
                self.full_trajs.append(self.active_trajs[i])
            else:
                occupied_map[int(next_xy[1]), int(next_xy[0])] = 1
                self.active_trajs[i].extend(next_time, next_xy)
                new_active_trajs.append(self.active_trajs[i])

        self.active_trajs = new_active_trajs

        if frame_path is None:
            return

        # generate the next sample candidates
        occupied_map_trans = scipy.ndimage.morphology.distance_transform_edt(
            1.0 - occupied_map
        )  # [H, W, 1]
        extracted_pts = self.generate_all_candidates(frame_path)  # [N, 2]

        # Get current active query points
        active_pts = self.get_cur_pos()  # shape: (N_active, 2)
        # Sample the candidates that are not occupied
        sample_map = occupied_map_trans[extracted_pts] > self.ratio
        non_active_candidates = extracted_pts[sample_map]
        # Combine active points and non-active candidates
        total_needed = 4096
        n_active = len(active_pts)
        n_non_active_needed = max(0, total_needed - n_active)
        if len(non_active_candidates) > n_non_active_needed:
            idx = np.random.choice(
                len(non_active_candidates), n_non_active_needed, replace=False
            )
            chosen_non_active = non_active_candidates[idx]
        else:
            chosen_non_active = non_active_candidates

        times = (np.ones(chosen_non_active.shape[0]) * next_time).astype(int)
        self.new_traj_all(times, chosen_non_active)
        self.sample_candidates = np.concatenate([active_pts, chosen_non_active], axis=0)

    def clear_active(self):
        for traj in self.active_trajs:
            self.full_trajs.append(traj)
        self.active_trajs = []


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
        self.invert_maps = {}

    def as_dict(self):
        """
        Returns a dict mapping traj_id -> Trajectory.as_dict().
        """
        output = {}
        for traj_id, traj in self.trajs.items():
            output[traj_id] = traj.as_dict()
        return output

    def build_invert_indexes(self):
        for traj_id, traj in tqdm(self.trajs.items(), desc="Building invert indexes"):
            for i in range(traj.length()):
                frame_id = traj.times[i]
                if frame_id not in self.invert_maps:
                    self.invert_maps[frame_id] = {}
                self.invert_maps[frame_id][traj_id] = i
