set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-29500}
export HYDRA_FULL_ERROR=1
export RAY_BACKEND_LOG_LEVEL=debug
export GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')


OUTPUT_DIR="./outputs/skywork_orz_ds_7b_ckpt"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=["./skywrok_data/Skywork-Orz-Rl-data/train_7b.pkl"] \
    data.val_files=["./skywrok_data/eval/aime24_x8.parquet","./skywrok_data/eval/aime25_x8.parquet"] \
    data.train_batch_size=16 \
    data.val_batch_size=13000 \
    data.max_prompt_length=2048 \
    data.max_response_length=16384 \
    actor_rollout_ref.model.path=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.entropy_coeff=0.005 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.adaptive_entropy.enabled=True \
    actor_rollout_ref.actor.adaptive_entropy.target_entropy=0.2 \
    actor_rollout_ref.actor.adaptive_entropy.max_ent_coef=0.5 \
    actor_rollout_ref.actor.adaptive_entropy.min_ent_coef=0 \
    actor_rollout_ref.actor.adaptive_entropy.delta_ent_coef=0.001 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=32 \
    actor_rollout_ref.rollout.n_val=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    reward_model.reward_manager=yr \
    trainer.critic_warmup=0 \
    trainer.rejection_sample=True \
    trainer.rejection_sample_multiplier=1 \
    trainer.logger=["console","wandb"] \
    trainer.project_name=yr_merge \
    trainer.experiment_name="skywork_orz_ds_7b" \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.stats_path=$OUTPUT_DIR/stats \
    trainer.stats_save_freq=1 \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=30 "${@:1}"