set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-29500}
export HYDRA_FULL_ERROR=1
export RAY_BACKEND_LOG_LEVEL=debug
export GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')
export LIVECODEBENCH_DATA_PATH=${LIVECODEBENCH_DATA_PATH:-./or1_data/eval/livecodebench/livecodebench_2408_2502}

MODEL_NAME=${MODEL_NAME:-Skywork/Skywork-OR1-32B-Preview}

# Evalation Aime24
python3 -m verl.trainer.main_generation \
    trainer.nnodes=$WORLD_SIZE \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    model.path=$MODEL_NAME \
    data.path=or1_data/eval/aime24.parquet \
    data.output_path=./outputs/evalation/Aime24_Avg32-Skywork_OR1_32B_Preview.pkl \
    data.n_samples=32 \
    data.batch_size=102400 \
    rollout.temperature=1.0 \
    rollout.response_length=32768 \
    rollout.top_k=-1 \
    rollout.top_p=1.0 \
    rollout.gpu_memory_utilization=0.8 \
    rollout.tensor_model_parallel_size=2

# Evalation Aime25
python3 -m verl.trainer.main_generation \
    trainer.nnodes=$WORLD_SIZE \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    model.path=$MODEL_NAME \
    data.path=or1_data/eval/aime25.parquet \
    data.output_path=./outputs/evalation/Aime25_Avg32-Skywork_OR1_32B_Preview.pkl \
    data.n_samples=32 \
    data.batch_size=102400 \
    rollout.temperature=1.0 \
    rollout.response_length=32768 \
    rollout.top_k=-1 \
    rollout.top_p=1.0 \
    rollout.gpu_memory_utilization=0.8 \
    rollout.tensor_model_parallel_size=2

# Evalation LiveCodeBench
python3 -m verl.trainer.main_generation \
    trainer.nnodes=$WORLD_SIZE \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    model.path=$MODEL_NAME \
    data.path=or1_data/eval/livecodebench/livecodebench_2408_2502.parquet \
    data.output_path=./outputs/evalation/LCB_Avg4-Skywork_OR1_32B_Preview.pkl \
    data.n_samples=4 \
    data.batch_size=102400 \
    rollout.temperature=1.0 \
    rollout.response_length=32768 \
    rollout.top_k=-1 \
    rollout.top_p=1.0 \
    rollout.gpu_memory_utilization=0.8 \
    rollout.tensor_model_parallel_size=2