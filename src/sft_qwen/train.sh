# --nnodes 1 --nproc_per_node 4 --master_port 25641
pwd=/data/lixubin/MiraTrainer
rm -rf $pwd/output/*

deepspeed --include localhost:1,2,3,4,5,6 $pwd/src/sft_qwen/examples/zero_nlp/qwen2_sft.py \
    --model_name_or_path /data/lixubin/models/Qwen/Qwen1.5-1.8B-Chat \
    --use_lora true \
    --use_deepspeed true \
    --data_path $pwd/data/sft/ \
    --bf16 true \
    --fp16 true \
    --output_dir $pwd/output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --report_to "tensorboard" \
    --learning_rate 4e-4 \
    --logging_steps 10 \
    --tf32 False \
    --model_max_length 2048

    # --save_strategy "steps" \
    # --save_steps 10 \ 
    # --save_steps 1000 \
