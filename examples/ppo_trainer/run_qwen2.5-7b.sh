set -x

export TP_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_VERSION=2.17.1
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=22
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_SPLIT_DATA_ON_QPS=1
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_IB_RETRY_CNT=13
export NCCL_NET_PLUGIN=none
export NCCL_SOCKET_IFNAME=eth0     #香港集群是bond0
export NCCL_DEBUG=WARN
export NCCL_CUMEM_ENABLE=0
export RAY_IGNORE_UNHANDLED_ERRORS=1
export VLLM_ATTENTION_BACKEND=XFORMERS


DATA_HOME=/mnt/2050data/yuhaoliu/projects/r1/datasets/verl_datasets

numina_train_path=$DATA_HOME/numina_hard/train.parquet
numina_val_path=$DATA_HOME/numina_hard/test.parquet
numina_test_path=$DATA_HOME/aime24_amc24_gaokao_math500/test.parquet


train_files="['$numina_train_path']"
test_files="['$numina_val_path', '$numina_test_path']"




export PYTHONPATH=$PYTHONPATH:/mnt/data/zifeng.cao/reasoning/descartes/verl
export WANDB_API_KEY=de14a48b0bc87c3ae1ca1ed23f7db5f37401367f

cd /mnt/data/zifeng.cao/reasoning/descartes/verl
pip install -i https://mirrors.cloud.tencent.com/pypi/simple -e .

# Add model and data configuration
MODEL_NAME_PATH_DICT=(
    # "DeepSeek-R1-Distill-Qwen-7B" "/mnt/data/models/DeepSeek-R1-Distill-Qwen-7B"
    # "DeepSeek-R1-Distill-Qwen-14B" "/mnt/data/models/DeepSeek-R1-Distill-Qwen-14B"
    # "DeepSeek-R1-Distill-Qwen-32B" "/mnt/data/models/DeepSeek-R1-Distill-Qwen-32B"
    # "Qwen2.5-0.5B-Instruct" "/mnt/data/zifeng.cao/reasoning/models/Qwen/Qwen2.5-0.5B-Instruct"
    "Qwen2.5-7B-Base" "/mnt/data/zifeng.cao/reasoning/models/Qwen/Qwen2.5-7B"
    # "Qwen2.5-32B-Base" "/mnt/data/models/Qwen2.5-32B"
    # "Qwen2.5-0.5B-Instruct" "/mnt/data/zifeng.cao/reasoning/models/Qwen/Qwen2.5-0.5B-Instruct"
    # "Qwen2.5-7B-Instruct" "/mnt/data/zifeng.cao/reasoning/models/Qwen/Qwen2.5-7B-Instruct"
    # "Qwen2.5-32B-Instruct" "/mnt/data/zifeng.cao/reasoning/models/Qwen/Qwen2.5-32B-Instruct"
    # "Qwen2.5-72B-Instruct" "/mnt/data/zifeng.cao/reasoning/models/Qwen/Qwen2.5-72B-Instruct"
)

ACTOR_MODEL_PATH=${MODEL_NAME_PATH_DICT[1]}
CRITIC_MODEL_PATH=${MODEL_NAME_PATH_DICT[1]}

EXP_NAME="numina_hard_${WORLD_SIZE}nodes_${MODEL_NAME_PATH_DICT[0]}"

# Check if local IP matches MASTER_ADDR
if [ "$(hostname)" == "${MASTER_ADDR}" ]; then
    # This is master node
    ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8

    python3 -m verl.trainer.main_ppo \
        data.train_files="$train_files" \
        data.val_files="$test_files" \
        data.train_batch_size=1024 \
        data.val_batch_size=6312 \
        data.max_prompt_length=1024 \
        data.max_response_length=4096 \
        actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
        actor_rollout_ref.actor.optim.lr=5e-7 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=256 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.grad_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.ref.fsdp_config.param_offload=False \
        critic.optim.lr=9e-6 \
        critic.model.use_remove_padding=True \
        critic.model.path=$CRITIC_MODEL_PATH \
        critic.model.enable_gradient_checkpointing=True \
        critic.ppo_micro_batch_size_per_gpu=8 \
        critic.model.fsdp_config.param_offload=False \
        critic.model.fsdp_config.grad_offload=False \
        critic.model.fsdp_config.optimizer_offload=False \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.critic_warmup=0 \
        trainer.logger=['console','wandb'] \
        trainer.project_name='verl_example' \
        trainer.experiment_name=$EXP_NAME \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=$WORLD_SIZE \
        trainer.save_freq=-1 \
        trainer.test_freq=25 \
        trainer.total_epochs=15 $@
else
    # This is worker node
    apt update
    apt install -y iputils-ping
    sleep 10
    ray start --address ${MASTER_ADDR}:6379 --num-gpus 8

    MAX_RETRIES=5  # 最大重试次数
    RETRY_DELAY=10  # 每次重试之间的延迟时间（单位：秒）
    PING_TIMEOUT=1  # 每次 ping 的超时时间（单位：秒）

    retry_count=0
    while true; do
        # 尝试 ping 主节点
        if ping -c 1 -W $PING_TIMEOUT $MASTER_ADDR; then
            echo "Master node $MASTER_ADDR is reachable."
            retry_count=0  # 如果成功ping通，重置重试次数
            sleep $RETRY_DELAY
        else
            echo "Master node $MASTER_ADDR is unreachable. Retry count: $retry_count"

            # 如果重试次数超过最大限制，则退出
            if [ $retry_count -ge $MAX_RETRIES ]; then
                echo "Exceeded maximum retries. Exiting..."
                break
            fi
            
            # 每次失败后增加重试计数，并停顿一段时间再试
            retry_count=$((retry_count + 1))
            echo "Sleeping for $RETRY_DELAY seconds before retrying..."
            sleep $RETRY_DELAY
        fi
    done
fi