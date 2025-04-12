# Copyright 2024 PRIME team and/or its affiliates
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

from .utils import check_correctness as apps_check_correctness
import json
import re
import traceback
from verl.utils.reward_score.livecodebench import lcb_compute_score, prepare_unit_test_data
import os, pickle

livecodebench_dir = '/mnt/data/rui.yan/projects/r1/datasets/verl_datasets/code/code_generation_lite/livecodebench_2408_2502'

def compute_score(completion, test_cases, task=None, continuous=False):
    # try to get code solution from completion. if the completion is pure code, this will not take effect.
    # solution = completion.split('```python')[-1].split('```')[0]


    if "question_id" in test_cases:
        try:
            if "</think>" in completion:
                solution_str = completion.split("</think>")[1]
            else:
                solution_str = completion

            benchmark = pickle.load(open(os.path.join(livecodebench_dir, "{}.pkl".format(test_cases["question_id"])), "rb"))
            custom_output = test_cases.copy()
            custom_output["output_list"] = [solution_str]
            return lcb_compute_score([custom_output], [benchmark]), None
        except:
            return False, None
    else:
        try:
            solutions = re.findall(r"```python\n(.*?)```", completion, re.DOTALL)
            if len(solutions) == 0:
                return False, None

            solution = solutions[-1]
            try:
                if not isinstance(test_cases, dict):
                    test_cases = json.loads(test_cases)
            except Exception as e:
                print(f"Error:{e}")
                return False, None

            # Complete check on all in-out pairs first. If there is no failure, per-sample test can be skipped.
            try:
                res, metadata = apps_check_correctness(in_outs=test_cases, generation=solution, timeout=5, debug=False)
                metadata = dict(enumerate(metadata))[0]
                success = all(map(lambda x: x == True, res))
                if success:
                    return success, metadata
            except Exception as e:
                success = False
                pass
            return success, None
            # test_cases_list = []
            # inputs = test_cases["inputs"]
            # outputs = test_cases["outputs"]
            # for i in range(len(inputs)):
            #     test_cases_list.append({"inputs": [inputs[i]], "outputs": [outputs[i]]})
        
            # if continuous:
            #     # per sample test: if continuous score is needed, test first 10 samples regardless of failures
            #     # do not test all samples cuz some problems have enormous test cases
            #     metadata_list = []
            #     res_list = []
            #     for test_case_id, test_case in enumerate(test_cases_list):
            #         res, metadata = apps_check_correctness(in_outs=test_case, generation=solution, timeout=5, debug=False)
            #         try:
            #             metadata = dict(enumerate(metadata))[0]  # metadata can be empty occasionally
            #         except Exception as e:
            #             metadata = {}
            #         metadata["test_case"] = {}
            #         metadata["test_case"]["input"] = str(test_case["inputs"][0])
            #         metadata["test_case"]["output"] = str(test_case["outputs"][0])
            #         metadata["test_case"]["res"] = str(res)
            #         metadata_list.append(metadata)
            #         res_list.extend(res)

            #         if test_case_id >= 9:
            #             break
            #     res_count = len(res_list) if len(res_list) > 0 else 1
            #     success = sum(map(lambda x: x == True, res_list)) / res_count
        except Exception as e:
            traceback.print_exc(10)
            return False, None
