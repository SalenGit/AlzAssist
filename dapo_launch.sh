# 获取 sft.py 的完整路径
SFT_PATH=$(python -c "import swift.cli.rlhf; print(swift.cli.rlhf.__file__)")

CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 ENABLE_AUDIO_OUTPUT=0 \
torchrun \
    --master_port 29500 \
    --nproc_per_node=4 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    $SFT_PATH \
    --rlhf_type grpo \
    --torch_dtype bfloat16 \
    --learning_rate 1e-5 \
    --external_plugins reward_format_accuracy.py \
    --reward_funcs format_accuracy soft_overlong \
    --num_generations 16 \
    --max_completion_length 2048 \
    --log_completions true \
    --model ./ckpt \
    --dataset  rlhf_dataset_replace_alz.json \
    --dataloader_num_workers 16 \
    --split_dataset_ratio 0 \
    --save_steps 100 \
    --save_total_limit 5 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --train_type lora \
    --gradient_checkpointing true \
    --lora_rank 32 \
    --target_modules 'all-linear' \
    --attn_impl flash_attn \
    --ddp_find_unused_parameters false \
    --output_dir 'output/dapo_noformat_v2' \
    --lora_bias 'all' \
    --freeze_vit true \
    --freeze_aligner true \
    --loss_type bnpo \
    --epsilon_high 0.28 \
    --dynamic_sample true \
    --max_resample_times 10 \
    --soft_cache_length 1024 \

    #--dataloader_persistent_workers true \
    #--use_vllm true \
    #--vllm_gpu_memory_utilization 0.5 \