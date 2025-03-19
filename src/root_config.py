from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from dacite import Config, from_dict
from omegaconf import OmegaConf

from libs.keypoint_detector import KeypointDetectorCfg
from libs.keypoint_matcher import KeypointMatcherCfg


@dataclass
class ColmapMapperCfg:
    max_num_models: int = 2
    # By default colmap does not generate a reconstruction
    # if less than 10 images are registered. Lower it to 3.
    min_model_size: int = 3


@dataclass
class RootCfg:
    base_dir: Path
    output_dir: Path

    keypoint_detector: KeypointDetectorCfg
    keypoint_matcher: KeypointMatcherCfg
    colmap_mapper: ColmapMapperCfg

    def to_yaml(self, path: str):
        config_dict = asdict(self)
        with open(path, "w") as file:
            yaml.dump(config_dict, file, default_flow_style=False, sort_keys=False)


TYPE_HOOKS = {
    Path: Path,
}


def load_typed_root_config(cfg_path: str) -> RootCfg:
    cfg = OmegaConf.load(cfg_path)
    return from_dict(
        RootCfg,
        OmegaConf.to_container(cfg),
        config=Config(type_hooks={**TYPE_HOOKS}),
    )
