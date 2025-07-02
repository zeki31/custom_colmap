import numpy as np
import scipy
import torch


class Trajectory(object):
    def __init__(self, start_time, start_xy, buffer_size=0):
        self.buffer_size = buffer_size
        self.times = []
        self.xys = []
        self.buffer_xys = []
        self.extend(start_time, start_xy)

    def extend(self, time, xy):
        self.times.append(time)
        self.buffer_xys.append(xy)
        if len(self.buffer_xys) > self.buffer_size:
            self.xys.append(self.buffer_xys[0])
            self.buffer_xys.pop(0)

    def length(self):
        return len(self.xys) + len(self.buffer_xys)

    def get_tail_location(self):
        if self.length() == 0:
            raise ValueError("Error!The trajectory is empty!")
        if len(self.buffer_xys) == 0:
            return self.xys[-1]
        else:
            return self.buffer_xys[-1]

    def clear_buffer(self):
        self.xys.extend(self.buffer_xys)
        self.buffer_xys = []

    def set_buffer_xy(self, index, xy):
        self.buffer_xys[index] = xy

    def as_dict(self):
        return {
            "frame_ids": self.times,
            "locations": self.xys,
        }


class IncrementalTrajectorySet(object):
    def __init__(self, total_length, img_h, img_w, sample_ratio):
        self.total_length = total_length
        self.ratio = sample_ratio
        self.h, self.w = img_h, img_w
        self.active_trajs = []
        self.full_trajs = []

        self.all_candidates = self.generate_all_candidates()
        self.sample_candidates = np.reshape(np.copy(self.all_candidates), (-1, 2))

    def generate_all_candidates(self):
        x, y = np.arange(0, self.w), np.arange(0, self.h)
        xx, yy = np.meshgrid(x, y)
        xys = np.stack([xx, yy], -1)
        s_xys = xys[:: self.ratio, :: self.ratio, :]
        return s_xys

    def new_traj_all(self, start_times, start_xys):
        for time, xy in zip(start_times, start_xys):
            # t = particlesfm.Trajectory(int(time), np.array(xy, dtype=np.float64).reshape(2, 1), buffer_size=0)
            t = Trajectory(time, xy, buffer_size=0)
            self.active_trajs.append(t)

    def get_cur_pos(self):
        # Get all the current traj positions
        cur_pos = []
        for i in range(len(self.active_trajs)):
            cur_pos.append(self.active_trajs[i].get_tail_location())
        return np.array(cur_pos)

    def extend_all(self, next_xys, next_time, flags):
        # Extend all the trajs
        assert len(next_xys) == len(self.active_trajs)
        assert len(flags) == len(next_xys)

        # also check the next sample candidates
        occupied_map = np.zeros((self.h, self.w, 1))

        self.new_active_trajs = []
        for i in range(len(flags)):
            next_xy, flag = next_xys[i], flags[i]
            if not flag:
                self.active_trajs[i].clear_buffer()
                self.full_trajs.append(self.active_trajs[i])
            else:
                occupied_map[int(next_xy[1]), int(next_xy[0])] = 1
                self.active_trajs[i].extend(next_time, next_xy)
                self.new_active_trajs.append(self.active_trajs[i])
        self.active_trajs = self.new_active_trajs

        # generate the next sample candidates
        occupied_map_trans = scipy.ndimage.morphology.distance_transform_edt(
            1.0 - occupied_map
        )
        sample_map = (occupied_map_trans > self.ratio)[:: self.ratio, :: self.ratio, 0]
        # Get current active query points
        active_pts = self.get_cur_pos()  # shape: (N_active, 2)
        non_active_candidates = np.copy(self.all_candidates[sample_map])
        # Remove any non-active candidates that are already in active_pts
        if len(active_pts) > 0 and len(non_active_candidates) > 0:
            # Use a set for fast lookup
            active_set = set(map(tuple, np.round(active_pts).astype(int)))
            non_active_candidates = np.array(
                [
                    pt
                    for pt in non_active_candidates
                    if tuple(np.round(pt).astype(int)) not in active_set
                ]
            )
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
        # Concatenate and pad if needed
        candidates = np.concatenate([active_pts, chosen_non_active], axis=0)
        if len(candidates) < total_needed:
            pad_count = total_needed - len(candidates)
            pad = (
                np.repeat(candidates[:1], pad_count, axis=0)
                if len(candidates) > 0
                else np.zeros((pad_count, 2))
            )
            candidates = np.concatenate([candidates, pad], axis=0)
        self.sample_candidates = candidates

    def clear_active(self):
        for traj in self.active_trajs:
            traj.clear_buffer()
            self.full_trajs.append(traj)
        self.active_trajs = []


def grid_sample(data, xy):
    # sample flow/feature value by xy indices
    # data: [C, H, W] of torch.tensor, xy: [N, 2], return: [N, C]
    data = data.unsqueeze(0)
    xy = torch.from_numpy(xy).float().to(data.device)
    xy = xy.unsqueeze(0).unsqueeze(0)
    H, W = data.shape[2], data.shape[3]
    xy[:, :, :, 0] /= (W - 1) / 2
    xy[:, :, :, 1] /= (H - 1) / 2
    xy -= 1
    out = torch.nn.functional.grid_sample(data, xy, align_corners=True)
    out = out.squeeze(0).squeeze(1).permute(1, 0).cpu().numpy()
    return out


class TrajectorySet:
    """
    A Python implementation of the C++ TrajectorySet.
    Relies on a Python Trajectory class with attributes:
      - times: list of frame IDs
      - xys: list of (x, y) locations
      - buffer_xys: list of buffered (x, y)
      - as_dict(): returns a dict with keys 'frame_ids', 'locations', 'labels'
    """

    def __init__(self, trajs: dict[int, Trajectory]):
        # trajs: mapping from traj_id to Trajectory instance
        self.trajs = {}
        # invert_maps: mapping from frame_id to {traj_id: index_in_trajectory}
        self.invert_maps = {}
        self.trajs = trajs

    def as_dict(self):
        """
        Returns a dict mapping traj_id -> Trajectory.as_dict().
        """
        output = {}
        for traj_id, traj in self.trajs.items():
            output[traj_id] = traj.as_dict()
        return output
