#!/bin/bash
steps_list=(4 8 16 32 64)
for steps in "${steps_list[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python gsm8k_run_sample_cond.py \
        --model_path exp_local/gsm8k/2025.02.28/110335_small_step_120000 \
        --dataset gsm8k \
        --steps $steps \
        --batch_size 128
    echo "steps: $steps done"
done
