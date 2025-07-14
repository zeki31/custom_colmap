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

## Running
```bash
python src/main.py -i config/main.yml -o wandb.mode=online
```
