#!/bin/bash
steps_list=(64)
for steps in "${steps_list[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python gsm8k_run_sample_cond.py \
        --model_path exp_local/gsm8k/2024.05.15/215827_medium_step_60000 \
        --dataset gsm8k \
        --steps $steps \
        --batch_size 1
    echo "steps: $steps done"
done

