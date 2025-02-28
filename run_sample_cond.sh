#!/bin/bash
steps_list=(4 8 16 32 64)
for steps in "${steps_list[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python gsm8k_run_sample_cond.py \
        --model_path exp_local/gsm8k/2025.02.27/094449 \
        --dataset gsm8k \
        --steps $steps \
        --batch_size 1
    echo "steps: $steps done"
done
