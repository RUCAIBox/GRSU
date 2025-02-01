import asyncio
import copy
import itertools
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Union, List, Dict

import aiofiles
import numpy as np

from src.infer.actor.base_actor import BaseActor
from src.infer.env import State, Node
from src.infer.user_model.base_user_model import BaseUserModel
from src.task.base_task import BaseTask


class BaseSearchAlgo(ABC):
    def __init__(
            self,
            task: BaseTask,
            init_state: State, actor_model: BaseActor, user_model: BaseUserModel = None,
            output_strategy: Union[List[dict[str, str]], dict[str, str]] = None, num_reward_on_output=5, num_candidate_for_reward=5,
            log_file_path=None,
    ):
        self.task = task

        self.id_iter = itertools.count()
        self.root_node = Node(id=next(self.id_iter), state=init_state)

        self.actor_model = actor_model
        self.user_model = user_model

        if isinstance(output_strategy, dict):
            self.output_strategy = [output_strategy]
        elif isinstance(output_strategy, list) or output_strategy is None:
            self.output_strategy = output_strategy
        else:
            raise NotImplementedError
        self.num_reward_on_output = num_reward_on_output
        self.num_candidate_for_reward = num_candidate_for_reward
        assert self.num_candidate_for_reward >= 1

        self.all_node_list = None
        self.selected_node_list = None

        self.log_file_path = log_file_path

    @abstractmethod
    async def __call__(self, *args, **kwargs):
        pass

    async def _expand(self, node: Node) -> List[Node]:
        child_node_info_list = await self.actor_model(node)
        child_node_list = [Node(id=next(self.id_iter), **child_node_info) for child_node_info in child_node_info_list]
        return child_node_list

    async def _evaluate_node(self, node: Node):
        reward = await self.user_model.evaluate_reward(node)
        node.add_reward(reward)

    async def _score_item(self, rec_item_list: List[str], rec_content: str) -> Dict[str, List[float]]:
        return await self.user_model.score_item(state=self.root_node.state, rec_item_list=rec_item_list, rec_content=rec_content, is_reward=False)

    async def _output(self, final_node_list):
        strategy2prediction = dict()

        # aggregate source
        strategy_source_set = set()
        for output_strategy in self.output_strategy:
            strategy_source = output_strategy['source']
            strategy_source_set.add(strategy_source)

        strategy_source2item_score_dict = dict()
        for strategy_source in strategy_source_set:
            item_score_dict = defaultdict(list)

            if strategy_source.startswith('item'):
                if 'raw' in strategy_source:
                    if 'final' in strategy_source:
                        # assert len(final_node_list) == 1
                        if len(final_node_list) > 1:
                            best_node = sorted(final_node_list, key=lambda x: x._reward, reverse=True)[0]
                        else:
                            best_node = final_node_list[0]
                    elif 'all' in strategy_source:
                        if len(self.all_node_list) > 1:
                            best_node = sorted(self.all_node_list, key=lambda x: x._reward, reverse=True)[0]
                        else:
                            best_node = self.all_node_list[0]
                    else:
                        raise NotImplementedError
                    item_list = best_node.state.rec_item_list
                    strategy_source2item_score_dict[strategy_source] = item_list
                    continue
                elif 'vote' in strategy_source:
                    if 'final' in strategy_source:
                        node_list = final_node_list
                    elif 'selected' in strategy_source:
                        node_list = self.selected_node_list
                    elif 'all' in strategy_source:
                        node_list = self.all_node_list
                    else:
                        raise NotImplementedError

                    for node in node_list:
                        for item in node.state.rec_item_list:
                            item_score_dict[item].append(1)

                elif 'reward' in strategy_source:
                    if 'all' in strategy_source:
                        node_list = self.all_node_list
                    elif 'selected' in strategy_source:
                        node_list = self.selected_node_list
                    else:
                        raise NotImplementedError

                    item_set = set()
                    item_cnt = defaultdict(int)
                    for node in node_list:
                        for item in node.state.rec_item_list:
                            item_set.add(item)
                        if node.state.rec_item_score is not None:
                            for item, score_list in node.state.rec_item_score.items():
                                item_score_dict[item].extend(score_list)
                                item_cnt[item] += len(score_list)

                    task_list = []
                    for item_to_reward in item_set:
                        score_num = self.num_reward_on_output - item_cnt[item_to_reward]
                        if score_num <= 0:
                            continue

                        other_item_set = copy.copy(item_set)
                        other_item_set.remove(item_to_reward)
                        other_item_list = list(other_item_set)
                        sample_other_item_list_set = set()

                        for _ in range(score_num):
                            if self.num_candidate_for_reward > 1:
                                sample_other_item_list = random.sample(other_item_list, self.num_candidate_for_reward - 1)
                                while frozenset(sample_other_item_list) in sample_other_item_list_set:
                                    sample_other_item_list = random.sample(other_item_list, self.num_candidate_for_reward - 1)
                                sample_other_item_list_set.add(frozenset(sample_other_item_list))

                                item_list_for_reward = copy.copy(sample_other_item_list)
                                item_list_for_reward.append(item_to_reward)
                                random.shuffle(item_list_for_reward)
                            else:
                                item_list_for_reward = [item_to_reward]

                            for item in item_list_for_reward:
                                item_cnt[item] += 1

                            item_info_list_for_reward = [f'{i + 1}. {self.task.get_item_info_str(item_name, self.task.item2info[item_name])}' for i, item_name in enumerate(item_list_for_reward)]
                            item_info_content_for_reward = '\n'.join(item_info_list_for_reward)
                            task = self._score_item(
                                rec_item_list=item_list_for_reward,
                                rec_content=item_info_content_for_reward
                            )
                            task_list.append(task)

                    for task in asyncio.as_completed(task_list):
                        cur_item2reward = await task
                        for item_name, reward_list in cur_item2reward.items():
                            item_score_dict[item_name].extend(reward_list)

                            # cur_item2reward = await self.world_model.score_item(
                            #     rec_item_list=item_list_for_reward,
                            #     rec_content=item_info_content_for_reward
                            # )
                            # for item_name, reward in cur_item2reward.items():
                            #     item_score_dict[item_name].extend(reward)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

            strategy_source2item_score_dict[strategy_source] = item_score_dict

        # aggregate method
        for output_strategy in self.output_strategy:
            item_score_dict = strategy_source2item_score_dict[output_strategy['source']]
            strategy_aggregation = output_strategy['aggregation']

            if strategy_aggregation == 'raw':
                output_strategy_str = f'{output_strategy["source"]}+{output_strategy["aggregation"]}'
                strategy2prediction[output_strategy_str] = item_score_dict
                continue

            elif strategy_aggregation == 'sum':
                item_score_list = sorted(list(item_score_dict.items()), key=lambda x: np.sum(x[1]), reverse=True)
            elif strategy_aggregation == 'mean':
                item_score_list = sorted(list(item_score_dict.items()), key=lambda x: np.mean(x[1]), reverse=True)
            else:
                raise NotImplementedError
            self.log(f'{output_strategy}\n{dict(item_score_list)}')

            pred_item_list = [item_score[0] for item_score in item_score_list]

            output_strategy_str = f'{output_strategy["source"]}+{output_strategy["aggregation"]}'
            strategy2prediction[output_strategy_str] = pred_item_list

        return strategy2prediction

    def log(self, log_str):
        if self.log_file_path is not None:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(log_str + '\n\n')
