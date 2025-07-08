// ParticleSfM
// Copyright (C) 2022  ByteDance Inc.

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "trajectory_base.h"

namespace particlesfm {

Trajectory::Trajectory(int time, V2D xy, VXD desc) {
    extend(time, xy, desc);
}


Trajectory::Trajectory(const std::vector<int>& times_, const std::vector<V2D>& xys_, const std::vector<VXD>& descs_) {
    times = times_;
    xys = xys_;
    descs = descs_;
}

Trajectory::Trajectory(py::dict dict) {
    if (dict.contains("frame_ids"))
        times = dict["frame_ids"].cast<std::vector<int>>();
    if (dict.contains("locations"))
        xys = dict["locations"].cast<std::vector<V2D>>();
}

py::dict Trajectory::as_dict() const{
    py::dict output;
    output["frame_ids"] = times;
    output["locations"] = xys;
    return output;
}

void Trajectory::extend(int time, V2D xy, VXD desc) {
    times.push_back(time);
    xys.push_back(xy);
    descs.push_back(desc);
}

int Trajectory::length() const {
    return int(xys.size());
}

V2D Trajectory::get_tail_location() const {
    if (length() == 0)
        throw std::runtime_error("Error! The trajectory is empty!");
    return xys.back();
}

std::map<int, py::dict> TrajectorySet::as_dict() const {
    std::map<int, py::dict> output;
    for (auto it = trajs.begin(); it != trajs.end(); ++it) {
        output[it->first] = it->second.as_dict();
    }
    return output;
}

TrajectorySet::TrajectorySet(std::map<int, py::dict> input) {
    for (auto it = input.begin(); it != input.end(); ++it) {
        trajs.insert(std::make_pair(it->first, Trajectory(it->second)));
    }
}

void TrajectorySet::build_invert_indexes() {
    for (auto it = trajs.begin(); it != trajs.end(); ++it) {
        auto traj = it->second;
        for (size_t i = 0; i < traj.length(); ++i) {
            int frame_id = traj.times[i];
            if (invert_maps.find(frame_id) == invert_maps.end())
                invert_maps.insert(std::make_pair(frame_id, std::map<int, size_t>()));
            invert_maps[frame_id].insert(std::make_pair(it->first, i));
        }
    }
}

} // namespace particlesfm
