steps_list=(64)
for steps in "${steps_list[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python gsm8k_run_sample_cond_mp_1.py \
        --model_path exp_local/gsm8k/2025.03.01/013931_dot_mp_small_60000 \
        --dataset gsm8k \
        --steps $steps \
        --batch_size 1
    echo "steps: $steps done"
done

