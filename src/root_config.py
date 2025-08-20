from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional

import yaml
from dacite import Config, from_dict
from omegaconf import OmegaConf

from src.mapping.mapper import MapperCfg
from src.matching import MatcherCfg
from src.matching.retriever import RetrieverCfg
from visualization import VisualizerCfg


@dataclass
class WandbCfg:
    entity: str
    project: str
    name: str
    group: str
    id: Optional[str]
    mode: Literal["online", "offline", "disabled"]


@dataclass
class RootCfg:
    base_dir: Path
    out_dir: Path
    wandb: WandbCfg

    prior_dir: Optional[Path]

    train: bool
    add_non_keyframe: bool
    viz: bool

    retriever: RetrieverCfg
    matcher: MatcherCfg
    mapper: MapperCfg
    visualizer: VisualizerCfg

    def to_yaml(self, path: Path):
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


def load_typed_root_config(cfg_path: Path, updates: Optional[list[str]]) -> RootCfg:
    cfg = OmegaConf.load(cfg_path)

    # Handle multiple config choices
    defaults = cfg.pop("defaults", [])
    cfg_dir = cfg_path.parent
    for item in defaults:
        for key, val in item.items():
            subcfg_path = cfg_dir / key / f"{val}.yml"
            subcfg = OmegaConf.load(subcfg_path)
            cfg[key] = subcfg

    if updates:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(updates))

    return from_dict(
        RootCfg,
        OmegaConf.to_container(cfg),
        config=Config(type_hooks={**TYPE_HOOKS}),
    )
