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
        save_dir: Path,
        retriever: Retriever,
    ):
        super().__init__(cfg, logger, device, paths, save_dir, retriever)

        self.tracker = Tracker(cfg.tracker, logger, save_dir)
        self.detector = KeypointDetector(
            cfg.keypoint_detector, logger, device, save_dir
        )
        self.matcher = KeypointMatcher(cfg.keypoint_matcher, logger, paths, save_dir)

    def match(self, image_paths: list[Path], feature_dir: Path) -> None:
        """Track points over frames in dynamic cameras and match keypoints in a fixed camera."""
        if (feature_dir / "matches.h5").exists():
            print("Already matched keypoints, skipping.")
            return

        start = time()

        self.tracker.track(image_paths, feature_dir)
        lap_tracking = time()
        print(f"Tracking completed in {(lap_tracking - start) // 60:.2f} minutes.")
        self.logger.summary["Tracking time (min)"] = (lap_tracking - start) // 60
        torch.cuda.empty_cache()
        gc.collect()

        self.detector.track_fixed(
            image_paths[: len(image_paths) // 4],
            feature_dir=feature_dir,
            viz=True,
        )

        print("Merging trajectories from all cameras...")
        full_trajs = []
        for cam_name in ["1_fixed", "2_dynA", "3_dynB", "4_dynC"]:
            trajs = np.load(
                feature_dir / cam_name / "full_trajs.npy", allow_pickle=True
            )
            full_trajs.extend(trajs)
        # Build TrajectorySet
        dict_trajs = {}
        for idx, traj in enumerate(full_trajs):
            dict_trajs[idx] = traj
        trajectories = TrajectorySet(dict_trajs)
        trajectories.build_invert_indexes()

        kpts_per_img = self.detector.register_keypoints(
            image_paths,
            feature_dir,
            trajectories,
            self.tracker.cfg.query,
            # viz=True,
        )
        torch.cuda.empty_cache()
        gc.collect()

        index_pairs = self.retriever.get_index_pairs(
            image_paths,
            "exhaustive_keyframe_excluding_same_view",
            self.cfg.tracker.window_len,
        )
        # TODO: parallelize this step
        traj_pairs = self.matcher.match_trajectories(
            image_paths,
            index_pairs,
            kpts_per_img,
            # viz=True,
        )
        torch.cuda.empty_cache()
        gc.collect()

        trajs = trajectories.trajs
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
        trajectories = TrajectorySet(trajs)
        trajectories.build_invert_indexes()

        print("Register keypoints again to update traj_ids...")
        kpts_per_img = self.detector.register_keypoints(
            image_paths,
            feature_dir,
            trajectories,
            self.tracker.cfg.query,
            # viz=True,
        )
        gc.collect()
        index_pairs = self.retriever.get_index_pairs(
            image_paths, "exhaustive_dynamic", self.cfg.tracker.window_len
        )
        # TODO: parallelize this step
        self.matcher.traj2match(
            image_paths,
            feature_dir,
            index_pairs,
            kpts_per_img,
            # viz=True,
        )
        gc.collect()

        end = time()
        self.logger.log({"Matching time (min)": (end - start) // 60})
