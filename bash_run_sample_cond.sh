steps_list=(1 2 4 8 16 32 64 128)
for steps in "${steps_list[@]}"; do
    python gsm8k_run_sample_cond.py \
        --model_path exp_local/gsm8k/2024.05.15/025053 \
        --dataset gsm8k \
        --steps $steps \
        --batch_size 128
    echo "steps: $steps done"
done

