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
uv pip install imageio[ffmpeg]

bash script/download_ckpts.sh
bash script/install_glomap.sh
```

Export `PYTHONPATH` to `.env` and `.venv/bin/activate`
```bash
bash ./setup_env.sh
```

## Running
```bash
python src/main.py --config_path config/main.yml
```
