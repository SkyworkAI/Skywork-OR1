# 文件名：taco_ray.py
import ray
import datasets
import json
import io
import contextlib
import warnings
from tqdm import tqdm
from verl.utils.reward_score.livecodebench import compute_score as compute_score_yr
from verl.workers.reward_manager.prime import parallel_compute_score_async
import asyncio

# 过滤掉 SyntaxWarning
# warnings.filterwarnings("ignore", category=SyntaxWarning)
# debug
ray.init(
    runtime_env={
        'env_vars': {
            'TOKENIZERS_PARALLELISM': 'true', 
            'NCCL_DEBUG': 'WARN', 
            "RAY_DEBUG":"1",
            "RAY_DEBUG_POST_MORTEM":"1"
            }
        },
    )
    
print("Ray 集群信息:", ray.cluster_resources())
def run(item, xx, yy):
    breakpoint()
    flag = False
    try:
        # 如果 "solutions" 是字符串，则解析 JSON，否则直接使用
        solutions = json.loads(item["solutions"]) if isinstance(item["solutions"], str) else item["solutions"]
    except Exception:
        solutions = []

    for cur_solution in solutions[:32]:
        # 屏蔽 compute_score_yr 产生的所有输出
        # with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        score = compute_score_yr("```python\n{}\n```".format(cur_solution), item['input_output'])[0]
        if score:
            flag = True
            break
    return flag


@ray.remote
def ray_prarllel(batchs):
    # 使用 asyncio.run 调用异步任务
    return asyncio.run(
        parallel_compute_score_async(
            run,
            [batchs[i:i+16] for i in range(0, len(batchs), 16)],
            [[1]*len(batchs[i:i+16]) for i in range(0, len(batchs), 16)],
            [[2]*len(batchs[i:i+16]) for i in range(0, len(batchs), 16)],
            num_processes=16
        )
    )

def chunk_list(lst, n):
    """将 lst 分割为每批 n 个样本的小批次"""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

if __name__ == "__main__":
    # 加载数据集，并过滤掉 solutions 为空的样本
    ds = datasets.load_dataset("/mnt/data/rui.yan/hf_dataset_model/datasets/TACO/ALL")
    ds_with_solutions = ds.filter(lambda x: x["solutions"] != "[]")

    # 选择 "test" 分片
    data_list = ds_with_solutions["test"].to_list()

    batch_size = 64
    batches = [data_list[i:i+batch_size] for i in  range(0, len(data_list), batch_size)]
    tasks = [ray_prarllel.remote(batch) for batch in batches]

    batch_results = []
    remaining = tasks.copy()
    pbar = tqdm(total=len(tasks), desc="Processing batches")
    while remaining:
        done, remaining = ray.wait(remaining, num_returns=len(remaining), timeout=1)
        if done:
            for r in done:
                batch_results.append(ray.get(r))
            pbar.update(len(done))
    pbar.close()

    # 展平结果，得到与 data_list 对应的布尔列表
    results = [flag for sublist in batch_results for flag in sublist]
    filtered_data = [item for item, flag in zip(data_list, results) if flag]

    # 将过滤后的数据转换回 Hugging Face 数据集，并保存为 JSONL 文件
    ds_filtered = datasets.Dataset.from_dict(filtered_data)
    ds_filtered.to_json("/mnt/data/rui.yan/hf_dataset_model/taco数据处理/test_ray.jsonl")
