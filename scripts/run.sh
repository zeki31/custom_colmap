python src/main.py -i config/main.yml -o base_dir=../datasets/sony_ai/valid/classroom2/talking wandb.name=classroom2_talking_ours_v2_60fps_winsize10
python src/main.py -i config/main.yml -o base_dir=../datasets/sony_ai/valid/classroom2/talking_cinematic wandb.name=classroom2_talking_cinematic_ours_v2_winsize2
# python src/main.py -i config/baseline.yml -o base_dir=../datasets/sony_ai/valid/classroom2/talking wandb.name=lane_running_baseline
# python src/main.py -i config/baseline.yml -o base_dir=../datasets/sony_ai/valid/classroom2/talking_cinematic wandb.name=lane_running_cinematic_baseline
