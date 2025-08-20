GROUP=$1

python src/main.py -i config/main.yml -u base_dir=../sony_ai/20250529/valid/lounge1/lunch wandb.name=lounge1_lunch wandb.group=$GROUP
python src/main.py -i config/main.yml -u base_dir=../sony_ai/20250529/valid/lane/running wandb.name=lane_running wandb.group=$GROUP
python src/main.py -i config/main.yml -u base_dir=../sony_ai/20250529/valid/parking/cycling wandb.name=parking_cycling wandb.group=$GROUP
python src/main.py -i config/main.yml -u base_dir=../sony_ai/20250529/valid/lounge2/talking wandb.name=lounge2_talking wandb.group=$GROUP
python src/main.py -i config/main.yml -u base_dir=../sony_ai/20250529/valid/cafe/serving wandb.name=cafe_serving wandb.group=$GROUP
python src/main.py -i config/main.yml -u base_dir=../sony_ai/20250529/valid/bench/talking wandb.name=bench_talking wandb.group=$GROUP
python src/main.py -i config/main.yml -u base_dir=../sony_ai/20250529/valid/classroom2/blackboard wandb.name=classroom2_blackboard wandb.group=$GROUP
python src/main.py -i config/main.yml -u base_dir=../sony_ai/20250529/valid/classroom2/play wandb.name=classroom2_play wandb.group=$GROUP
python src/main.py -i config/main.yml -u base_dir=../sony_ai/20250529/valid/classroom2/talking wandb.name=classroom2_talking wandb.group=$GROUP
# python src/main.py -i config/main.yml -u base_dir=../sony_ai/20250529/valid/lounge1/leaving wandb.name=lounge1_leaving wandb.group=$GROUP
# python src/main.py -i config/main.yml -u base_dir=../sony_ai/20250529/valid/parking/walking wandb.name=parking_walking wandb.group=$GROUP
# python src/main.py -i config/main.yml -u base_dir=../sony_ai/20250529/valid/lane/walking1 wandb.name=lane_walking1 wandb.group=$GROUP
# python src/main.py -i config/main.yml -u base_dir=../sony_ai/20250529/valid/lane/walking2 wandb.name=lane_walking2 wandb.group=$GROUP
# python src/main.py -i config/main.yml -u base_dir=../sony_ai/20250529/valid/classroom1/meeting wandb.name=classroom1_meeting wandb.group=$GROUP
# python src/main.py -i config/main.yml -u base_dir=../sony_ai/20250529/valid/forum/talking wandb.name=forum_talking wandb.group=$GROUP
