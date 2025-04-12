import json
import os
import torch
from glob import glob
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import argparse

def load_and_merge_models(model_path: str, ckpt_path: str, save_path: str, world_size: int = 64):
    """
    加载基础模型并合并多个checkpoint的权重
    
    Args:
        model_path: 基础模型路径
        ckpt_path: checkpoint文件所在目录
        save_path: 最终保存模型的路径
    """

    # 依次加载每个checkpoint的权重
    state_dict = defaultdict(list)
    for rank in range(world_size):
        ckpt_file = os.path.join(ckpt_path, f"actor/model_world_size_{world_size}_rank_{rank}.pt")
        print(f"loading {ckpt_file.split('/')[-1]}")
        this_state_dict = torch.load(ckpt_file)
        for key, value in this_state_dict.items():
            state_dict[key].append(value.to_local())
    
    for key in state_dict:
        state_dict[key] = torch.cat(state_dict[key], dim=0)

    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_config(config)
    print(f'Saving actor checkpoint to {save_path}')
    model.load_state_dict(state_dict)
    model.save_pretrained(save_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(save_path)

def main():
    parser = argparse.ArgumentParser(description='合并多个checkpoint权重并转换为HuggingFace模型格式')
    parser.add_argument('--global_step', type=int, required=True, help='训练的全局步数')
    parser.add_argument('--base_path', type=str, required=True)
    parser.add_argument('--world_size', type=int, default=64, help='分布式训练的world_size')
    parser.add_argument('--save_path', type=str, default=None)

    args = parser.parse_args()
    
    # 构建完整的checkpoint路径
    ckpt_path = os.path.join(args.base_path, f"global_step_{args.global_step}")
    # 读取config获取原始模型路径
    config_path = os.path.join(ckpt_path, "actor/huggingface/config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    model_path = config["_name_or_path"]
    # 设置保存路径
    if args.save_path is None:
        save_path = os.path.join(ckpt_path, "huggingface")
    else:
        save_path = args.save_path
    # 执行模型合并
    load_and_merge_models(model_path, ckpt_path, save_path, args.world_size)

if __name__ == "__main__":
    main()





