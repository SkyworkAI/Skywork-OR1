set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-29500}
export HYDRA_FULL_ERROR=1
export RAY_BACKEND_LOG_LEVEL=debug
export GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')


TASK_NAME=skywork_orz_math_7b
OUTPUT_PATH=./outputs/evalation/$TASK_NAME.pkl

MODEL_PATH=Skywork/Skywork-Orz-Math-7B
DATA_PTH=skywrok_data/eval/aime24_x8.parquet

python3 -m verl.trainer.main_generation \
    trainer.nnodes=$WORLD_SIZE \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    model.path=$MODEL_PATH \
    data.path=$DATA_PTH \
    data.output_path=$OUTPUT_PATH \
    data.n_samples=1 \
    data.batch_size=102400 \
    rollout.temperature=0.6 \
    rollout.response_length=32768 \
    rollout.top_k=-1 \
    rollout.top_p=0.95 \
    rollout.gpu_memory_utilization=0.8 \
    rollout.tensor_model_parallel_size=2
