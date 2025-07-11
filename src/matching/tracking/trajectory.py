from pathlib import Path
from typing import Literal, Optional

import h5py
import kornia as K
import numpy as np
import scipy
import torch
from jaxtyping import Bool, Float
from numpy.typing import NDArray
from tqdm import tqdm

from src.colmap.database import image_ids_to_pair_id
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
        sample_ratio: int,
        device: torch.device,
        init_frame_path: Path,
        query: Literal["grid", "aliked"],
    ):
        self.total_length = total_length
        self.ratio = sample_ratio
        self.h, self.w = img_h, img_w
        self.active_trajs = []
        self.full_trajs = []
        self.query = query

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

        if self.query == "grid":
            self.all_candidates = self.generate_grid_candidates()
            self.sample_candidates = np.reshape(np.copy(self.all_candidates), (-1, 2))

        elif self.query == "aliked":
            (
                self.sample_candidates,
                self.candidate_desc,
            ) = self.generate_aliked_candidates(init_frame_path)

    def _load_torch_image(self, file_name: Path | str, device=torch.device("cpu")):
        """Loads an image and adds batch dimension"""
        img = K.io.load_image(file_name, K.io.ImageLoadType.RGB32, device=device)[
            None, ...
        ]
        return img

    def generate_grid_candidates(self) -> Float[NDArray, "h_sampled w_sampled 2"]:
        """Grid sampling of the image."""
        x, y = np.arange(0, self.w), np.arange(0, self.h)
        xx, yy = np.meshgrid(x, y)
        xys = np.stack([xx, yy], -1, dtype=np.float32)
        s_xys = xys[:: self.ratio, :: self.ratio, :]
        return s_xys

    def generate_aliked_candidates(
        self, frame_path: Path
    ) -> tuple[Float[NDArray, "N 2"], Float[NDArray, "N D"]]:
        with torch.inference_mode():
            image = self._load_torch_image(frame_path, device=self.device).to(
                self.dtype
            )
            features = self.extractor.extract(image)
        kpts = features["keypoints"].detach().cpu().numpy().squeeze(0)  # shape: (N, 2)
        # valid_cond = (
        #     (kpts[:, 0] > 0)
        #     & (kpts[:, 0] < self.w - 1)
        #     & (kpts[:, 1] > 0)
        #     & (kpts[:, 1] < self.h - 1)
        # )
        # kpts = kpts[valid_cond].copy()  # shape: (N_valid, 2)
        descs = features["descriptors"].detach().cpu().numpy().squeeze(0)
        return kpts, descs

    def new_traj_all(self, start_times, start_xys, start_desc=None):
        if self.query == "grid":
            for time, xy in zip(start_times, start_xys):
                t = Trajectory(int(time), xy)
                self.active_trajs.append(t)

        elif self.query == "aliked":
            start_desc = start_desc if start_desc is not None else self.candidate_desc
            for time, xy, desc in zip(start_times, start_xys, start_desc):
                t = Trajectory(int(time), xy, desc)
                self.active_trajs.append(t)

    def get_cur_pos(self):
        # Get all the current traj positions
        cur_pos = []
        for i in range(len(self.active_trajs)):
            cur_pos.append(self.active_trajs[i].get_tail_location())
        return np.array(cur_pos)

    def get_cur_pos_desc(
        self,
    ) -> tuple[Float[NDArray, "N_curr 2"], Float[NDArray, "N_curr D"]]:
        # Get all the current traj positions
        cur_pos = []
        cur_desc = []
        for i in range(len(self.active_trajs)):
            cur_pos.append(self.active_trajs[i].get_tail_location())
            cur_desc.append(self.active_trajs[i].descs[-1])
        return np.array(cur_pos), np.array(cur_desc)

    def extend_all(
        self,
        next_xys: Float[NDArray, "N 2"],
        next_time: int,
        flags: Bool[NDArray, " N"],
        next_descs: Optional[Float[NDArray, "N D"]] = None,
        frame_path: Path = None,
    ):
        # print(f"Extending {len(self.active_trajs)} active trajectories with {len(next_xys)} new points at time {next_time}.")
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
            next_desc = next_descs[i] if next_descs is not None else None
            # flag = flag if int(next_xy[0]) < self.w and int(next_xy[1]) < self.h else False
            if not flag:
                self.full_trajs.append(self.active_trajs[i])
            else:
                occupied_map[int(next_xy[1]), int(next_xy[0])] = 1
                self.active_trajs[i].extend(next_time, next_xy, next_desc)
                new_active_trajs.append(self.active_trajs[i])

        self.active_trajs = new_active_trajs

        if frame_path is None:
            return

        # generate the next sample candidates
        occupied_map_trans = scipy.ndimage.morphology.distance_transform_edt(
            1.0 - occupied_map
        )  # [H, W, 1]
        if self.query == "grid":
            sample_map = (occupied_map_trans > self.ratio)[
                :: self.ratio, :: self.ratio, 0
            ]
            # Get current active query points
            active_pts = self.get_cur_pos()  # shape: (N_active, 2)
            non_active_candidates = np.copy(self.all_candidates[sample_map])
            # Combine active points and non-active candidates
            total_needed = 3600
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
            self.sample_candidates = np.concatenate(
                [active_pts, chosen_non_active], axis=0
            )

        elif self.query == "aliked":
            extracted_pts, extracted_descs = self.generate_aliked_candidates(
                frame_path
            )  # [N, 2]

            # Get current active query points
            (
                active_pts,
                active_pts_desc,
            ) = self.get_cur_pos_desc()  # shape: (N_active, 2)
            # Sample the candidates that are not occupied
            xs = extracted_pts[:, 0].astype(int)
            ys = extracted_pts[:, 1].astype(int)
            # If occupied_map_trans has shape (H, W, 1), squeeze the last dimension
            distances = occupied_map_trans[ys, xs].squeeze()  # shape: (N,)
            sample_map = distances > self.ratio
            non_active_candidates = extracted_pts[sample_map]
            non_active_candidates_desc = extracted_descs[sample_map]
            # print(f"Active points: {active_pts.shape}, Non-active candidates: {non_active_candidates.shape}")

            # Combine active points and non-active candidates
            total_needed = 4096
            n_active = len(active_pts)
            n_non_active_needed = max(0, total_needed - n_active)
            if len(non_active_candidates) > n_non_active_needed:
                idx = np.random.choice(
                    len(non_active_candidates), n_non_active_needed, replace=False
                )
                chosen_non_active = non_active_candidates[idx]
                chosen_non_active_desc = non_active_candidates_desc[idx]
            else:
                chosen_non_active = non_active_candidates
                chosen_non_active_desc = non_active_candidates_desc

            times = (np.ones(chosen_non_active.shape[0]) * next_time).astype(int)
            self.new_traj_all(times, chosen_non_active, chosen_non_active_desc)
            self.sample_candidates = np.concatenate(
                [active_pts, chosen_non_active], axis=0
            )
            self.candidate_desc = np.concatenate(
                [active_pts_desc, chosen_non_active_desc], axis=0
            )

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

    def build_match_indexes(
        self,
        feature_dir: Path,
        mask_imgs: list[NDArray],
        kpts_indices: dict[int, dict[int, int]],
        n_frames: int,
        i_proc: int,
    ):
        """Save matches in a h5 file for each image."""
        with h5py.File((feature_dir / f"matches_{i_proc}.h5"), mode="w") as f_matches:
            added = set()
            for idx1, trajs_dict in tqdm(
                self.invert_maps.items(), desc="Building match indexes"
            ):
                kpts_indices_idx1 = kpts_indices[idx1]
                matches_dict = {}
                for traj_id in sorted(trajs_dict):
                    traj = self.trajs[traj_id]
                    kpts_indices_idx1_traj_id = kpts_indices_idx1[traj_id]
                    y, x = traj.xys[0].astype(int)
                    if mask_imgs[idx1][x, y] > 0:
                        continue

                    for idx2 in traj.times:
                        if idx1 == idx2:
                            continue
                        if idx2 not in matches_dict:
                            matches_dict[idx2] = []
                        matches_dict[idx2].append(
                            (kpts_indices_idx1_traj_id, kpts_indices[idx2][traj_id])
                        )

                for idx2, matches in matches_dict.items():
                    # Store the matches in the group of one image
                    if len(matches) < 15:
                        continue

                    # idx1 = idx1 if idx1 < n_frames else 0
                    pair_id = image_ids_to_pair_id(idx1, idx2)
                    if pair_id in added:
                        print(
                            f"Warning: Duplicate matches for {idx1} and {idx2}. Skipping."
                        )
                        continue
                    group = f_matches.require_group(str(idx1))
                    group.create_dataset(str(idx2), data=np.array(matches))
                    added.add(pair_id)
