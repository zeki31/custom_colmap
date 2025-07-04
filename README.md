# custom_colmap
Run structure from motion using COLMAP

## Setup

Initialize venv
```
uv venv
```

Install dependencies
```
uv sync
uv pip install imageio[ffmpeg]

bash script/download_ckpts.sh
bash script/install_glomap.sh
```

Export `PYTHONPATH` to `.env` and `.venv/bin/activate`
```
./setup_env.sh
```

## Running
```
python src/main.py --config_path config/main.yml
```
