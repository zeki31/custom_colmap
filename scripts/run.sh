python src/main.py -i config/main.yml -o base_dir=../datasets/sony_ai/20250529/valid/classroom2/blackboard wandb.name=classroom2_blackboard_ours
python src/main.py -i config/main.yml -o base_dir=../datasets/sony_ai/20250529/valid/classroom2/blackboard_cinematic wandb.name=classroom2_blackboard_cinematic_ours
python src/main.py -i config/baseline.yml -o base_dir=../datasets/sony_ai/20250529/valid/classroom2/blackboard wandb.name=classroom2_blackboard_baseline
python src/main.py -i config/baseline.yml -o base_dir=../datasets/sony_ai/20250529/valid/classroom2/blackboard_cinematic wandb.name=classroom2_blackboard_cinematic_baseline
