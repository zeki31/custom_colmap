import gc
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Literal

import numpy as np
import torch
import wandb
from tqdm import tqdm

from src.matching.matcher import Matcher
from src.matching.retriever import Retriever
from src.matching.sparse.keypoint_detector import KeypointDetector, KeypointDetectorCfg
from src.matching.sparse.keypoint_matcher import KeypointMatcher, KeypointMatcherCfg
from src.matching.tracking.tracker import Tracker, TrackerCfg
from src.matching.tracking.trajectory import TrajectorySet
from src.matching.tracking.union_find import UnionFind

mp.set_start_method("spawn", force=True)


@dataclass
class MatcherTrackingCfg:
    name: Literal["tracking"]
    tracker: TrackerCfg
    keypoint_detector: KeypointDetectorCfg
    keypoint_matcher: KeypointMatcherCfg


class MatcherTracking(Matcher[MatcherTrackingCfg]):
    def __init__(
        self,
        cfg: MatcherTrackingCfg,
        logger: wandb.sdk.wandb_run.Run,
        device: torch.device,
        paths: list[Path],
        feature_dir: Path,
        save_dir: Path,
        retriever: Retriever,
    ):
        super().__init__(cfg, logger, device, paths, feature_dir, save_dir, retriever)

        self.tracker = Tracker(cfg.tracker, logger, save_dir)
        self.detector = KeypointDetector(
            cfg.keypoint_detector, logger, device, save_dir
        )
        self.matcher = KeypointMatcher(
            cfg.keypoint_matcher, logger, paths, feature_dir, save_dir
        )
        self.stride = cfg.tracker.window_len - cfg.tracker.overlap

    def match(self) -> None:
        """Track points over frames in dynamic cameras and match keypoints in a fixed camera."""
        if (self.feature_dir / "matches_0.h5").exists():
            print("Already matched keypoints, skipping.")
            return

        start = time()

        self.tracker.track(self.paths, self.feature_dir)
        lap_tracking = time()
        print(f"Tracking completed in {(lap_tracking - start) // 60:.2f} minutes.")
        self.logger.summary["Tracking time (min)"] = (lap_tracking - start) // 60
        torch.cuda.empty_cache()
        gc.collect()

        print("Merging trajectories from all cameras...")
        full_trajs_aliked = []
        full_trajs_grid = []
        for cam_name in ["2_dynA", "3_dynB", "4_dynC"]:
            trajs_aliked = np.load(
                self.feature_dir / cam_name / "full_trajs_aliked.npy", allow_pickle=True
            )
            full_trajs_aliked.extend(trajs_aliked)

            trajs_grid = np.load(
                self.feature_dir / cam_name / "full_trajs_grid.npy", allow_pickle=True
            )
            full_trajs_grid.extend(trajs_grid)

        if full_trajs_aliked:
            trajs_fixed = self.detector.track_fixed(
                self.paths[: len(self.paths) // 4],
                feature_dir=self.feature_dir,
                viz=True,
            )
            full_trajs_aliked.extend(trajs_fixed)

            dict_trajs_aliked = {}
            for idx, traj in enumerate(full_trajs_aliked):
                dict_trajs_aliked[idx] = traj
            trajectories = TrajectorySet(dict_trajs_aliked)
            trajectories.build_invert_indexes()

            kpts_per_img = self.detector.register_keypoints(
                self.paths,
                self.feature_dir,
                trajectories,
                only_aliked=True,
                viz=self.cfg.keypoint_detector.viz,
            )
            torch.cuda.empty_cache()
            gc.collect()

            index_pairs = self.retriever.get_index_pairs(
                self.paths,
                "exhaustive_keyframe_excluding_same_view",
                self.stride,
            )
            traj_pairs_list = self.matcher.multiprocess(
                self.matcher.match_trajectories,
                index_pairs,
                2,
                (self.feature_dir / "matches_0.h5").exists(),
                kpts_per_img,
            )
            traj_pairs = {pair for pairs_set in traj_pairs_list for pair in pairs_set}
            torch.cuda.empty_cache()
            gc.collect()

            trajs = trajectories.trajs
            max_id = max(trajs.keys())
            uf = UnionFind(len(trajs))
            for traj_id1, traj_id2 in tqdm(traj_pairs, desc="Extending trajectories"):
                uf.union(traj_id1, traj_id2)
            for traj_id in tqdm(trajs.copy(), desc="Merging trajectories"):
                root_traj_id = uf.root(traj_id)
                if root_traj_id == traj_id:
                    continue
                traj = trajs.pop(traj_id)
                trajs[root_traj_id].xys.extend(traj.xys)
                trajs[root_traj_id].descs.extend(traj.descs)
                trajs[root_traj_id].times.extend(traj.times)
            # Add grid trajectories
            for idx, traj in enumerate(full_trajs_grid, start=max_id + 1):
                trajs[idx] = traj
            trajectories = TrajectorySet(trajs)
        else:
            dict_trajs = {}
            for idx, traj in enumerate(full_trajs_grid):
                dict_trajs[idx] = traj
            trajectories = TrajectorySet(dict_trajs)

        trajectories.build_invert_indexes()

        print("Register keypoints again to update traj_ids...")
        kpts_per_img = self.detector.register_keypoints(
            self.paths,
            self.feature_dir,
            trajectories,
            only_aliked=False,
            viz=self.cfg.keypoint_detector.viz,
        )
        gc.collect()
        if self.cfg.tracker.query == "grid":
            index_pairs = self.retriever.get_index_pairs(
                self.paths,
                "frame",
            )
            self.matcher.traj2npy(
                index_pairs,
                self.feature_dir,
                kpts_per_img,
            )
            exit()
        else:
            index_pairs = self.retriever.get_index_pairs(
                self.paths,
                "exhaustive_dynamic",
                self.stride,
            )
            _ = self.matcher.multiprocess(
                self.matcher.traj2match,
                index_pairs,
                4,
                (self.feature_dir / "matches_0.h5").exists(),
                kpts_per_img,
            )
        gc.collect()

        end = time()
        self.logger.log({"Matching time (min)": (end - start) // 60})
