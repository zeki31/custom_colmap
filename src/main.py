import argparse
import gc
import warnings
from pathlib import Path

import torch
import wandb
from jaxtyping import install_import_hook

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.colmap.importer import COLMAPImporter
    from src.mapping.mapper import Mapper
    from src.matching import get_matcher
    from src.matching.retriever import Retriever
    from src.root_config import load_typed_root_config

warnings.simplefilter("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--cfg_path", type=Path)
    parser.add_argument("-o", "--overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_typed_root_config(args.cfg_path, args.overrides)

    save_dir = cfg.out_dir / cfg.wandb.group / cfg.wandb.name
    save_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(save_dir / "config.yaml")

    feature_dir = save_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    database_path = feature_dir / "colmap.db"
    if database_path.exists():
        database_path.unlink()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=f"{cfg.wandb.group}/{cfg.wandb.name}",
        dir=save_dir,
        config=cfg.to_dict(),
        group=cfg.wandb.group,
        id=cfg.wandb.id,
        mode=cfg.wandb.mode,
        resume="allow",
    ) as logger:
        retriever = Retriever(cfg.retriever, logger)
        image_paths = retriever.get_image_paths(cfg.base_dir)

        matcher = get_matcher(
            cfg.matcher, logger, device, image_paths, feature_dir, save_dir, retriever
        )
        matcher.match()
        gc.collect()

        # Import keypoint distances of matches into colmap for RANSAC
        importer = COLMAPImporter(logger, retriever)
        importer.import_into_colmap(
            database_path, feature_dir, image_paths, cfg.matcher.name, cfg.base_dir
        )

        mapper = Mapper(cfg.mapper, logger)
        mapper.map(
            database_path,
            cfg.base_dir,
            save_dir,
        )


if __name__ == "__main__":
    main()
