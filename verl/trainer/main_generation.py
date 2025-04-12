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
Generate responses given a dataset of prompts
"""
import csv
import ray
import numpy as np
import hydra
import os
from tabulate import tabulate
from functools import partial

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from verl.utils.model import compute_position_id_with_mask

import pandas as pd

from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.workers.reward_manager.yr_code import parallel_compute_score
from verl.utils.reward_score.livecodebench import compute_score  as compute_score_yr
import math
import json

@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    local_path = copy_local_path_from_hdfs(config.model.path)
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)
    # tokenizer.model_max_length = 32768
    # Check if output file already exists
    if os.path.exists(config.data.output_path):
        print(f"Output file {config.data.output_path} already exists. Skipping generation and proceeding to evaluation.")
        if config.data.output_path.endswith(".pkl"):
            dataset = pd.read_pickle(config.data.output_path)
            if not isinstance(dataset, pd.core.frame.DataFrame):
                dataset = pd.DataFrame(dataset)
        else:
            dataset = pd.read_parquet(config.data.output_path)
    else:
        if config.rollout.temperature == 0.:
            assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'

        # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)

        if config.data.path.endswith(".pkl"):
            dataset = pd.read_pickle(config.data.path)
            if not isinstance(dataset, pd.core.frame.DataFrame):
                dataset = pd.DataFrame(dataset)
        elif config.data.path.endswith(".jsonl"):
            dataset = [json.loads(x) for x in open(config.data.path)]
            if not isinstance(dataset, pd.core.frame.DataFrame):
                dataset = pd.DataFrame(dataset)
        else:
            dataset = pd.read_parquet(config.data.path)
            
        chat_lst = dataset[config.data.prompt_key].tolist()

        chat_lst = [(chat.tolist() if not isinstance(chat, list) else chat) for chat in chat_lst]

        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='rollout')
        resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
        wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
        wg.init_model()

        total_samples = len(dataset)
        # real_batch_size = data.batch['input_ids'].shape[0]
        config_batch_size = config.data.batch_size
        dp_size = wg.world_size // config.rollout.tensor_model_parallel_size
        num_batch = (total_samples // config_batch_size) + 1
        output_lst = []  # We'll reshape at the end

        for batch_idx in range(num_batch):
            print(f'[{batch_idx+1}/{num_batch}] Start to process.')
            batch_chat_lst = chat_lst[batch_idx * config_batch_size:(batch_idx + 1) * config_batch_size]
            
            # Repeat the batch n_samples times
            repeated_chat_lst = []
            for chat in batch_chat_lst:
                repeated_chat_lst.extend([chat] * config.data.n_samples)
            
            inputs = tokenizer.apply_chat_template(repeated_chat_lst,
                                                 add_generation_prompt=True,
                                                 padding=True,
                                                 truncation=True,
                                                 max_length=config.rollout.prompt_length,
                                                 return_tensors='pt',
                                                 return_dict=True,
                                                 tokenize=True)
            
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            position_ids = compute_position_id_with_mask(attention_mask)

            batch_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}

            data = DataProto.from_dict(batch_dict)
            real_batch_size = data.batch['input_ids'].shape[0]
            
            if real_batch_size % dp_size != 0 or real_batch_size % wg.world_size != 0:
                lcm_value = math.lcm(dp_size, wg.world_size)
                adjusted_batch_size = (real_batch_size // lcm_value + 1) * lcm_value
                dummy_data_size = adjusted_batch_size - real_batch_size
                dummy_data = data[:dummy_data_size]
                data = DataProto.concat([data, dummy_data])
                print(
                    f'dp_size {dp_size} is not divisible by real_batch_size {real_batch_size}, add {dummy_data_size} dummy data'
                )

            batch_size = data.batch['input_ids'].shape[0]
            assert batch_size % dp_size == 0, f'batch_size {batch_size} is not divisible by dp_size {dp_size}'

            print(f'[{batch_idx+1}/{num_batch}] Start to generate.')
            
            # Generate all samples at once
            print(len(data.batch['input_ids']))
            output = wg.generate_sequences(data)
            # Remove dummy data
            output = output[:real_batch_size]
            output_text = tokenizer.batch_decode(output.batch['input_ids'][:, -config.rollout.response_length:],
                                               skip_special_tokens=False)

            # Remove padding
            pad_token = tokenizer.pad_token
            output_text_unpad = []
            for text in output_text:
                output_text_unpad.append(text.replace(pad_token, ''))

            output_lst.extend(output_text_unpad)

        # Reshape output_lst from (total_samples,) to (n_data, n_samples)
        total_samples = len(output_lst)
        n_data = total_samples // config.data.n_samples
        output_lst = np.array(output_lst).reshape(n_data, config.data.n_samples).tolist()

        # Add to the data frame
        dataset['responses'] = output_lst

        # Write to a new parquet
        output_dir = os.path.dirname(config.data.output_path)
        makedirs(output_dir, exist_ok=True)
        dataset.to_pickle(config.data.output_path)
    
    output_dir = os.path.dirname(config.data.output_path)
    # Compute evaluation metrics
    prompts = dataset[config.data.prompt_key]

    reward_model_data = dataset[config.data.reward_model_key]
    reward_model_data = [x['ground_truth'] for x in reward_model_data for _ in range(config.data.n_samples)]
    dataset_name = os.path.basename(config.data.path)
    row_data = {
        'model_path': config.model.path,
        'dataset': dataset_name,
        'ex_name': os.path.basename(config.data.output_path),
    }
    
    compute_score_yr_with_args = partial(compute_score_yr, is_binary_reward=False)
    scores = parallel_compute_score(
        compute_score_yr_with_args,
        [x for xx in dataset['responses'] for x in xx],
        reward_model_data,
        [1]*len(reward_model_data),
        max_workers=48,
        timeout=6,
    )

    scores = np.array(scores).reshape(-1, config.data.n_samples)
    # metadata = [metadata[idx:idx+config.data.n_samples] for idx in range(0, len(metadata), config.data.n_samples)]

    pass_at_n = (scores.max(-1) == 1).mean()
    reward = scores.mean()
    pass_at_1 = (scores[:,0] == 1).mean()
    pass_at_1_avg_sample = (scores[:,:] == 1).mean()
    
    row_data.update({
        f'{config.rollout.response_length//1024}K_Pass@1': pass_at_1,
        f'{config.rollout.response_length//1024}K_Pass@1(avg_{config.data.n_samples})': pass_at_1_avg_sample,
        f'{config.rollout.response_length//1024}K_Pass@{config.data.n_samples}': pass_at_n,
        f'{config.rollout.response_length//1024}K_Reward@1': reward,
    })

    dataset["score"] = scores.tolist()
    # dataset["metadata"] = metadata
    dataset.to_pickle(config.data.output_path)

    csv_path = os.path.join(output_dir, 'pass.csv')
    # Check if file exists
    file_exists = os.path.isfile(csv_path)
    
    # Write to CSV
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

    # Convert the row data into a list of lists format for tabulate
    table_data = [[k, v] for k, v in row_data.items()]
    
    # Print table
    print(tabulate(table_data, headers=['Metric', 'Value'], tablefmt='grid'))


    

if __name__ == '__main__':
    main()