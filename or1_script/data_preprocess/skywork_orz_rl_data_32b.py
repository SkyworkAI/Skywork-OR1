# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse
import json
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./skywrok_data/Skywork-Orz-Rl-data')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'Skywork/Skywork-Orz-Rl-data'

    dataset = datasets.load_dataset(data_source)
    print("\n\nOriginal dataset: ", "\n", dataset)

    def process_ground_truth(item):
        if "reward_model" in item and "ground_truth" in item["reward_model"]:
            try:
                item["reward_model"]["ground_truth"] = json.loads(item["reward_model"]["ground_truth"])
            except:
                pass
        return item
    
    dataset= dataset.map(process_ground_truth)
    
    def filter_fn(example):  

        if 'extra_info' not in example or 'model_difficulty' not in example['extra_info']:
            return False 
        difficulty = example['extra_info']['model_difficulty'].get('DeepSeek-R1-Distill-Qwen-32B')
        if difficulty is None:
            return False
        if difficulty < 1 or difficulty > 15:
            return False
        return True
    dataset = dataset.filter(filter_fn)
    print("\n\nFiltered dataset: ", "\n", dataset)

    data_list = []
    for key in dataset:
        data_list.extend([item for item in dataset[key]])
    
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    os.makedirs(local_dir, exist_ok=True)
    with open(os.path.join(local_dir, 'train_32b.pkl'), 'wb') as f:
        pickle.dump(data_list, f)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)