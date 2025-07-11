# custom_colmap
Run structure from motion using COLMAP

## Setup

Initialize venv
```bash
uv venv
```

Install dependencies
```bash
uv sync

bash scripts/download_ckpts.sh
bash scripts/install_glomap.sh
```

Export `PYTHONPATH` to `.env` and `.venv/bin/activate`
```bash
bash ./setup_env.sh
```

## Running
```bash
python src/main.py -i config/main.yml -o wandb.mode=online
```
