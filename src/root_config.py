from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional

import yaml
from dacite import Config, from_dict
from omegaconf import OmegaConf

from libs.keypoint_detector import KeypointDetectorCfg
from libs.keypoint_matcher import KeypointMatcherCfg
from libs.retriever import RetrieverCfg


@dataclass
class MapperCfg:
    name: Literal["colmap", "glomap"] = "colmap"
    max_num_models: int = 2
    # By default colmap does not generate a reconstruction
    # if less than 10 images are registered. Lower it to 3.
    min_model_size: int = 3


@dataclass
class RootCfg:
    base_dir: Path
    output_dir: Path

    prior_dir: Optional[Path]
    retriever: RetrieverCfg
    keypoint_detector: KeypointDetectorCfg
    keypoint_matcher: KeypointMatcherCfg
    mapper: MapperCfg

    def to_yaml(self, path: str):
        """Save the configuration to a YAML file."""
        with open(path, "w") as file:
            yaml.dump(
                self.to_dict(),
                file,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary, converting Paths to strings."""

        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj

        return convert(asdict(self))


TYPE_HOOKS = {
    Path: Path,
}


def load_typed_root_config(cfg_path: Path, overrides: Optional[list[str]]) -> RootCfg:
    cfg = OmegaConf.load(cfg_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    return from_dict(
        RootCfg,
        OmegaConf.to_container(cfg),
        config=Config(type_hooks={**TYPE_HOOKS}),
    )
