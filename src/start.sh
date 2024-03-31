for arg in "$@"
do
    echo "Script Task: $arg"

    if [ $arg = "pt" ]; then
        ########## Clean output director ##########
        rm -rf /data/lixubin/MiraTrainer/src/output/*
        ########## Accelerate launch PT ##########
        accelerate launch \
            --config_file /data/lixubin/MiraTrainer/src/tuner/accelerate_config.yaml \
            ./tuner/run_pt.py   
    fi

    if [ $arg = "sft" ]; then
        ########## Clean output director ##########
        rm -rf /data/lixubin/MiraTrainer/src/output/*
        ########## Accelerate launch SFT ##########
        accelerate launch \
            --config_file /data/lixubin/MiraTrainer/src/tuner/accelerate_config.yaml \
            /data/lixubin/MiraTrainer/src/tuner/run_sft.py   
    fi

    if [ $arg = "dpo" ]; then
        ########## Clean output director ##########
        rm -rf /data/lixubin/MiraTrainer/src/output/*
        ########## Accelerate launch DPO ##########
        accelerate launch \
            --config_file /data/lixubin/MiraTrainer/src/tuner/accelerate_config.yaml \
            /data/lixubin/MiraTrainer/src/tuner/run_dpo.py

    fi

    if [ $arg = "cli" ]; then
        ########## Test model with LoRA ##########
        CUDA_VISIBLE_DEVICES=1,2 \
        python /data/lixubin/MiraTrainer/src/demo/qwen_cli_demo.py \
            -c /data/lixubin/MiraTrainer/src/output
    fi

done
