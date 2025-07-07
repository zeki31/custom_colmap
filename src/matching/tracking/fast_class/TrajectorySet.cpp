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
