set -x

# prepare pre-trained model ckpt
# huggingface-cli download deepseek-ai/deepseek-llm-7b-chat --local-dir $HOME/models/deepseek-llm-7b-chat

# ``actor_rollout_ref.rollout.tensor_model_parallel_size`` in theory could be different from
# ``**.megatron.tensor_model_parallel_size``

# the config file used: verl/trainer/main_ppo/config/ppo_megatron_trainer.yaml

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

export PYTHONPATH=$PYTHONPATH:/mnt/data/zhoubian/verl/Megatron-LM-verl
export WANDB_API_KEY=ac8bb2df0a6cccc290b84c5445f52327c8215a5d

pip install -i https://mirrors.cloud.tencent.com/pypi/simple -e .

# Check if local IP matches MASTER_ADDR
if [ "$(hostname)" == "${MASTER_ADDR}" ]; then
    # This is master node
    ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8
    
    python3 -m verl.trainer.main_ppo --config-path=config \
            --config-name='ppo_megatron_trainer.yaml'\
            data.train_files=/mnt/data/zhoubian/data/gsm8k/train.parquet \
            data.val_files=/mnt/data/zhoubian/data/gsm8k/test.parquet \
            data.train_batch_size=1024 \
            data.val_batch_size=1312 \
            data.max_prompt_length=512 \
            data.max_response_length=512 \
            actor_rollout_ref.model.path=/mnt/data/models/deepseek-llm-7b-chat/deepseek-llm-7b-chat \
            actor_rollout_ref.actor.optim.lr=2e-6 \
            actor_rollout_ref.actor.ppo_mini_batch_size=256 \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
            actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
            actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
            actor_rollout_ref.rollout.name=vllm \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
            actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4 \
            critic.optim.lr=2e-5 \
            critic.model.path=/mnt/data/models/deepseek-llm-7b-chat/deepseek-llm-7b-chat \
            critic.model.enable_gradient_checkpointing=False \
            critic.ppo_micro_batch_size_per_gpu=4 \
            critic.megatron.tensor_model_parallel_size=4 \
            algorithm.kl_ctrl.kl_coef=0.001 \
            trainer.critic_warmup=0 \
            trainer.logger=['console','wandb'] \
            trainer.project_name='verl_megatron_gsm8k_examples' \
            trainer.experiment_name='deepseek_llm_7b_function_rm' \
            trainer.n_gpus_per_node=8 \
            trainer.nnodes=2 \
            trainer.save_freq=-1 \
            trainer.total_epochs=15 \
            +trainer.val_before_train=False $@
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