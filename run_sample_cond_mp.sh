steps_list=(1)
for steps in "${steps_list[@]}"; do
    python gsm8k_run_sample_cond.py \
        --model_path exp_local/gsm8k/2024.05.17/071425_medium_step_60000 \
        --dataset gsm8k \
        --steps $steps \
        --batch_size 1
    echo "steps: $steps done"
done

