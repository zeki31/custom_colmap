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
```
Export `PYTHONPATH` to `.env` and `.venv/bin/activate`
```
./setup_env.sh
```

## Running
```
cd src
python main.py --config_path config/main.yml
```
