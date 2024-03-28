# CUDA_VISIBLE_DEVICES=1,2,3,4 python ./tuner/run_pt.py

rm -rf /data/lixubin/MiraTrainer/src/output/*

accelerate launch \
    --config_file /data/lixubin/MiraTrainer/src/tuner/accelerate_config.yaml \
    ./tuner/run_pt.py   