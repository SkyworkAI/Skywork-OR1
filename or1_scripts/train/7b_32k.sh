#!/bin/bash
set -ex

export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-29500}
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1

# Entropy Config
ENTROPY_COEFF=0.0
USE_ADAPTIVE_ENT=True
TGT_ENTROPY=0.2
MAX_ENT_COEF=0.005
MIN_ENT_COEF=0
DELTA_ENT_COEF=0.0001

ROLLOUT_BATCH_SIZE=256
PPO_MINI_BATCH=256
MAX_PROMPT_LENGTH=2048
RES_LENGTH=32768
GROUP_SIZE=16
N_VAL_SAMPLES=8

TRAIN_TEMPERATURE=1.0

TP=1
SP=1
MAX_TOKEN_LEN=$(((RES_LENGTH + MAX_PROMPT_LENGTH + 1000) / SP))

# Your Model Path
MODEL_PATH=${MODEL_PATH:-}
CODE_PATH=${CODE_PATH:-}
if [ -z "$MODEL_PATH" ]; then
    echo "MODEL_PATH is not set"
    exit 1
fi
if [ -z "$CODE_PATH" ]; then
    echo "CODE_PATH is not set"
    exit 1
fi

# Since math queries are much more than code queries, we duplicate the math data when mixing the datasets
train_files="[\"$CODE_PATH/or1_data/train/train_7b_code.pkl\",\"$CODE_PATH/or1_data/train/train_7b_code.pkl\",\"$CODE_PATH/or1_data/train/train_7b_math.pkl\"]"
test_files="[\"$CODE_PATH/or1_data/eval/aime24.parquet\",\"$CODE_PATH/or1_data/eval/aime25.parquet\"]"

PROJECT_NAME=skywork-or1-train

EXP_NAME=7B_L$(($RES_LENGTH / 1024))k
MODEL_NAME=$(basename $MODEL_PATH)
EXP_NAME=$EXP_NAME-${MODEL_NAME}-bs${ROLLOUT_BATCH_SIZE}-minibs${ROLLOUT_BATCH_SIZE}-gs${GROUP_SIZE}-tgt${TGT_ENTROPY}-temp${TRAIN_TEMPERATURE}-${WORLD_SIZE}nodes
SAVE_DIR=$CODE_PATH/verl_ckpt/$PROJECT_NAME/$EXP_NAME
SAVE_STATS_DIR=${SAVE_DIR}/stats
mkdir -p $SAVE_DIR
mkdir -p $SAVE_STATS_DIR

export RAY_DEBUG=1

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.train_batch_size=$ROLLOUT_BATCH_SIZE \
    data.val_batch_size=13000 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$RES_LENGTH \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKEN_LEN \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.adaptive_entropy.enabled=$USE_ADAPTIVE_ENT \
    actor_rollout_ref.actor.adaptive_entropy.target_entropy=${TGT_ENTROPY} \
    actor_rollout_ref.actor.adaptive_entropy.max_ent_coef=${MAX_ENT_COEF} \
    actor_rollout_ref.actor.adaptive_entropy.min_ent_coef=${MIN_ENT_COEF} \
    actor_rollout_ref.actor.adaptive_entropy.delta_ent_coef=${DELTA_ENT_COEF} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=$TRAIN_TEMPERATURE \
    actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=$GROUP_SIZE \
    actor_rollout_ref.rollout.n_val=$N_VAL_SAMPLES \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    reward_model.reward_manager=yr \
    trainer.critic_warmup=0 \
    trainer.rejection_sample=True \
    trainer.rejection_sample_multiplier=1 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$WORLD_SIZE \
    trainer.save_freq=20 \
    trainer.test_freq=20\
    trainer.stats_path=$SAVE_STATS_DIR \
    trainer.stats_save_freq=1 \
    trainer.default_local_dir=$SAVE_DIR \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=30 "${@:1}"
    