steps_list=(64)
for steps in "${steps_list[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python gsm8k_run_sample_cond_mp_1.py \
        --model_path exp_local/gsm8k/2024.05.20/071828_small_step_60000_mp \
        --dataset gsm8k \
        --steps $steps \
        --batch_size 1
    echo "steps: $steps done"
done

