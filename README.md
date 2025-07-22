# custom_colmap
Run structure from motion using COLMAP

## Setup

Install COLMAP/GLOMAP
```bash
bash scripts/compile.sh
```

Build environment
```bash
# Initialize venv
uv venv
# Install dependencies
uv sync
# Export `PYTHONPATH` to `.env` and `.venv/bin/activate`
bash ./setup_env.sh
# Download checkpoints
bash scripts/download_ckpts.sh
```

## Data Preparetion
```bash
python scripts/data_preprocess/cut_invalid_frames.py -b folder_path_containing_synchronized_videos -f 415 -l 1680
python scripts/data_preprocess/video2frames.py -b folder_path_containing_videos
```

## Running
```bash
python src/main.py -i config/main.yml -o wandb.mode=online
```
