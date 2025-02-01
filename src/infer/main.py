import argparse
import asyncio
import importlib
import json
import multiprocessing
import os
from collections import defaultdict
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm.asyncio import tqdm_asyncio

from src.infer.actor.rec_model import rec_model_with_retriever
from src.infer.env import State
from src.task.base_task import BaseTask, TaskData


class Agent:
    def __init__(
            self,
            task_param: dict,
            search_algo_param: dict,
            actor_param: dict,
            rec_model_param: dict,
            model_param: List[dict],
            user_model_param: dict = None,
            # reward_model_param: dict = None,
            log_file_dir: str = None,
            result_file_dir: str = None,
            cache_file_dir: str = None,
            n_process: int = 1
    ):
        self.log_file_dir = log_file_dir
        if self.log_file_dir is not None:
            # if os.path.exists(self.log_file_dir):
            #     shutil.rmtree(self.log_file_dir)
            os.makedirs(self.log_file_dir, exist_ok=True)

        self.result_file_dir = result_file_dir
        if self.result_file_dir is not None:
            # if os.path.exists(self.result_file_dir):
            #     shutil.rmtree(self.result_file_dir)
            os.makedirs(self.result_file_dir, exist_ok=True)

        self.cache_file_dir = cache_file_dir

        # task
        task_module = importlib.import_module(f'src.task.{task_param["type"]}')
        self.task: BaseTask = getattr(task_module, 'Task')(**task_param["config"])

        self.n_process = min(min(n_process, os.cpu_count()), len(self.task.dataset))

        # model
        self.model_param = model_param
        # self.model_dict = dict()
        # for param in model_param:
        #     model_module = importlib.import_module(f'src.model.{param["type"]}')
        #     model = getattr(model_module, 'Model')(**param["config"])
        #     self.model_dict[model.model_name] = model

        # rec_model
        rec_model_module = importlib.import_module(f'src.infer.actor.rec_model.{rec_model_param["type"]}')
        self.rec_model_class = getattr(rec_model_module, 'RecModel')
        self.rec_model_config = rec_model_param.get("config", dict())
        self.rec_model_config['task'] = self.task
        # self.rec_model_config['model_dict'] = self.model_dict

        # user_model
        self.user_model_class = None
        if user_model_param is not None:
            user_model_module = importlib.import_module(f'src.infer.user_model.{user_model_param["type"]}')
            self.user_model_class = getattr(user_model_module, 'UserModel')
            self.user_model_config = user_model_param.get("config", dict())
            self.user_model_config['k'] = self.rec_model_config['k']
            # self.user_model_config['model_dict'] = self.model_dict

        # actor
        actor_module = importlib.import_module(f'src.infer.actor.{actor_param["type"]}')
        self.actor_class = getattr(actor_module, 'Actor')
        self.actor_config = actor_param.get("config", dict())
        self.actor_config['task'] = self.task
        # # reward_model
        # self.reward_model_class = None
        # if reward_model_param is not None:
        #     reward_model_module = importlib.import_module(f'src.infer.reward_model.{reward_model_param["type"]}')
        #     self.reward_model_class = getattr(reward_model_module, 'RewardModel')
        #     self.reward_model_config = reward_model_param.get("config", dict())

        search_algo_module = importlib.import_module(f'src.infer.search_algo.{search_algo_param["type"]}')
        self.search_algo_class = getattr(search_algo_module, 'SearchAlgo')
        self.search_algo_config = search_algo_param.get("config", dict())
        self.search_algo_config['task'] = self.task

        if self.search_algo_config['num_beams'] > 1:
            self.actor_config['init_n'] = self.actor_config['n'] ** 2

    # async def _log_json(self, log_str, log_file_path=None):
    #     if log_file_path is not None:
    #         async with aiofiles.open(log_file_path, 'w', encoding='utf-8') as f:
    #             await f.write(log_str)

    def _log_json(self, log_str, log_file_path=None):
        if log_file_path is not None:
            with open(log_file_path, 'w', encoding='utf-8') as f:
                f.write(log_str)

    # async def _log_str(self, log_str, log_file_path=None, end='\n\n'):
    #     if log_file_path is not None:
    #         async with aiofiles.open(log_file_path, 'w', encoding='utf-8') as f:
    #             await f.write(f'{log_str}{end}')

    def _log_str(self, log_str, log_file_path=None, end='\n\n'):
        if log_file_path is not None:
            with open(log_file_path, 'w', encoding='utf-8') as f:
                f.write(f'{log_str}{end}')

    # async def _load_json(self, file_path):
    #     async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
    #         contents = await f.read()
    #         return json.loads(contents)

    def _load_json(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    async def search(self, data, exist_cache=False, model_dict=None):
        # log
        log_file_path = None
        if self.log_file_dir is not None:
            log_file_path = os.path.join(self.log_file_dir, f'{data.conv_id}-{data.turn_id}.log')
            if os.path.exists(log_file_path):
                os.remove(log_file_path)
            self._log_str(json.dumps(data, ensure_ascii=False), log_file_path=log_file_path)

        init_state = State(
            conv_id=data.conv_id, turn_id=data.turn_id, dialog_history_list=data.dialog_history_list,
            user_id=data.user_id, user_history_list=data.user_history_list
        )

        # rec_model
        rec_model = self.rec_model_class(
            **self.rec_model_config, model_dict=model_dict, log_file_path=log_file_path
        )

        # user_model
        user_model = None
        if self.user_model_class is not None:
            user_model = self.user_model_class(**self.user_model_config, model_dict=model_dict, log_file_path=log_file_path)

        # actor
        actor_model = self.actor_class(rec_model=rec_model, user_model=user_model, **self.actor_config)

        search_algo = self.search_algo_class(
            **self.search_algo_config,
            init_state=init_state, actor_model=actor_model, user_model=user_model,
            log_file_path=log_file_path
        )

        all_node_list = None
        if exist_cache is True:
            cache_file_path = os.path.join(self.cache_file_dir, f'{data.conv_id}-{data.turn_id}.json')
            cache_content = self._load_json(cache_file_path)
            all_node_list = cache_content['all_node']

        prediction_dict, output_dict = await search_algo(all_node_list)

        result_dict = dict(**data, **output_dict)
        all_metric_dict = defaultdict(lambda: defaultdict(list))
        if prediction_dict is not None:
            cur_result_dict = defaultdict(dict)
            for output_strategy, pred_item_list in prediction_dict.items():
                # if self.prediction_need_clean is True:
                #     pred_item_list = self.task.clean_prediction(pred_item_list)
                # cur_result_dict[output_strategy]['pred_item_list'] = pred_item_list

                pred_item_list, metric_dict = self.task.cal_metric(pred_item_list, data.rec_label_list)

                cur_result_dict[output_strategy]['pred_item_list'] = pred_item_list
                cur_result_dict[output_strategy]['metric'] = metric_dict

                for metric, val in metric_dict.items():
                    all_metric_dict[output_strategy][metric].append(val)

            result_dict = dict(**data, prediction=cur_result_dict)
            if self.cache_file_dir is None:
                result_dict = dict(**result_dict, **output_dict)

        if self.result_file_dir is not None:
            result_file_path = os.path.join(self.result_file_dir, f'{data.conv_id}-{data.turn_id}.json')
            self._log_json(log_str=json.dumps(result_dict, ensure_ascii=False), log_file_path=result_file_path)

        all_metric_dict = dict(all_metric_dict)

        return all_metric_dict

    def worker_process(self, data_list, exist_cache):
        model_dict = dict()
        for param in self.model_param:
            model_module = importlib.import_module(f'src.model.{param["type"]}')

            if 'max_parallel_requests' not in param["config"]:
                param["config"]['max_parallel_requests'] = 256
            param["config"]['n_process'] = self.n_process

            model = getattr(model_module, 'Model')(**param["config"])
            model_dict[model.model_name] = model

        task_list = []
        for data in data_list:
            data = TaskData(**data)
            task = self.search(data, exist_cache, model_dict)
            task_list.append(task)

        metric_dict_list = []
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        for task in tqdm_asyncio.as_completed(task_list):
            metric_dict = loop.run_until_complete(task)
            metric_dict_list.append(metric_dict)

        return metric_dict_list

    def _set_item_embeddings(self):
        model_dict = dict()
        for param in self.model_param:
            model_module = importlib.import_module(f'src.model.{param["type"]}')
            if 'max_parallel_requests' not in param["config"]:
                param["config"]['max_parallel_requests'] = 256
            model = getattr(model_module, 'Model')(**param["config"])
            model_dict[model.model_name] = model

        rec_model = self.rec_model_class(**self.rec_model_config, model_dict=model_dict)
        retrieve_model = rec_model.retrieve_model

        task_list = []
        for i in range(len(self.task.id2item)):
            item_info_str = self.task.get_item_info_str_for_embedding(self.task.id2item[i])
            task_list.append(retrieve_model.get_embedding([item_info_str]))

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        task_output_list = loop.run_until_complete(tqdm_asyncio.gather(*task_list))
        loop.close()

        item_embeddings = [item_embedding[0] for item_embedding in task_output_list]
        item_embeddings = F.normalize(torch.tensor(item_embeddings), p=2, dim=-1)
        self.task.item_embeddings = item_embeddings

    def run(self):
        all_metric_dict = defaultdict(lambda: defaultdict(list))

        exist_cache = False
        if self.cache_file_dir is not None:
            exist_cache = os.path.exists(os.path.join(self.cache_file_dir, 'COMPLETED'))
        print(f'exist cache: {exist_cache}')

        if exist_cache is False and self.rec_model_class == rec_model_with_retriever.RecModel:
            self._set_item_embeddings()

        # task_list = []
        # for batch_data in self.task_dataloader:
        #     for data in batch_data:
        #         data = TaskData(**data)
        #         task = self.search(data, exist_cache)
        #         task_list.append(task)
        #         # task_list.append(pool.apply(self.search, (data, all_metric_dict, exist_cache)))
        #
        # loop = asyncio.get_event_loop()
        # loop.run_until_complete(tqdm_asyncio.gather(*task_list, total=len(task_list)))

        data_list = [data for data in self.task.dataset]
        slice_idx = np.linspace(0, len(data_list), self.n_process + 1).astype('int')

        with multiprocessing.Pool(self.n_process) as p:
            task_list = []
            for process_id in range(self.n_process):
                start, end = slice_idx[process_id], slice_idx[process_id + 1]

                batch_data_list = data_list[start: end]
                task = p.apply_async(self.worker_process, args=(batch_data_list, exist_cache))
                task_list.append(task)

            for task in task_list:
                for metric_dict in task.get():
                    for output_strategy, metric_val_dict in metric_dict.items():
                        for metric, val in metric_val_dict.items():
                            all_metric_dict[output_strategy][metric].append(val)

        # async with aiomultiprocess.Pool(processes=self.n_process) as pool:
        #     task_list = [pool.apply(self.search, (TaskData(**data), exist_cache)) for data in self.task.dataset]
        #     for task in tqdm_asyncio.as_completed(task_list):
        #         metric_dict = await task
        #         for output_strategy, cur_metric_dict in metric_dict.items():
        #             for metric, val in cur_metric_dict.items():
        #                 all_metric_dict[output_strategy][metric].append(val)

        complete_file_path = os.path.join(self.result_file_dir, 'COMPLETED')
        self._log_str('COMPLETED', log_file_path=complete_file_path, end='')

        # if len(all_metric_dict) > 0:
        #     for output_strategy, metric_dict in all_metric_dict.items():
        #         for metric, val_list in metric_dict.items():
        #             all_metric_dict[output_strategy][metric] = np.mean(val_list)
        #         print(output_strategy)
        #         print(json.dumps(metric_dict))

        for output_strategy, metric_dict in all_metric_dict.items():
            for metric, val_list in metric_dict.items():
                all_metric_dict[output_strategy][metric] = np.mean(val_list)
            print(output_strategy)
            print(json.dumps(metric_dict))

        if self.result_file_dir is not None:
            result_file_path = os.path.join(os.path.dirname(self.result_file_dir), f'{os.path.basename(self.result_file_dir)}.json')
            self._log_json(log_str=json.dumps(all_metric_dict, ensure_ascii=False), log_file_path=result_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_path', required=True)
    args = parser.parse_args()

    with open(args.config_file_path) as f:
        config = yaml.safe_load(f)
    print(json.dumps(config))

    agent = Agent(**config)
    agent.run()
