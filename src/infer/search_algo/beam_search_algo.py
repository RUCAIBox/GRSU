import asyncio
import json
from collections import defaultdict

from src.infer.env import Node, State
from src.infer.search_algo.base_search_algo import BaseSearchAlgo


class SearchAlgo(BaseSearchAlgo):
    def __init__(self, num_beams=1, max_depth=1, reward_on_step_end=True, *args, **kwargs):
        super(SearchAlgo, self).__init__(*args, **kwargs)

        self.num_beams = num_beams
        self.max_depth = max_depth

        self.reward_on_step_end = reward_on_step_end

    async def __call__(self, all_node_list=None):
        if all_node_list is None:

            pre_node_list = [self.root_node]
            self.all_node_list = []
            self.selected_node_list = []
            self.log(f'root node\n{json.dumps(self.root_node.to_dict())}')

            # try:
            for depth in range(self.max_depth):
                cur_node_list = []
                task_list = [self._expand(node) for node in pre_node_list]
                for task in asyncio.as_completed(task_list):
                    child_node_list = await task
                    cur_node_list.extend(child_node_list)

                assert len(cur_node_list) >= self.num_beams, print(self.root_node.to_dict())
                self.all_node_list.extend(cur_node_list)

                if self.reward_on_step_end is False:
                    pre_node_list = cur_node_list
                else:
                    task_list = [self._evaluate_node(node) for node in cur_node_list]
                    await asyncio.gather(*task_list)

                    pre_node_list = sorted(cur_node_list, key=lambda node: node.reward, reverse=True)[:self.num_beams]

                child_node_str = f"id: {', '.join([str(node.id) for node in cur_node_list])}"
                self.log(f'depth {depth}\n\nexpand\n{child_node_str}')

                pre_node_str = f"id: {', '.join([str(node.id) for node in pre_node_list])}"
                self.log(f'select\n{pre_node_str}')

                self.selected_node_list.extend(pre_node_list)

            final_node_list = pre_node_list
            for node in final_node_list:
                node.is_terminal = True
        else:
            self.all_node_list = []
            self.selected_node_list = []
            final_node_list = []
            depth2node_list = defaultdict(list)

            for node_info in all_node_list:
                depth = node_info['depth']
                if depth <= self.max_depth:
                    node = Node(
                        id=node_info['id'], depth=node_info['depth'], reward=node_info['reward'],
                        state=State(**node_info['state']), action=node_info['action'], transition=node_info['transition'],
                    )
                    node.is_terminal = node_info['is_terminal']

                    self.all_node_list.append(node)

                    if node.is_terminal is True:
                        final_node_list.append(node)

                    depth2node_list[depth].append(node)

            for depth, node_list in depth2node_list.items():
                node_list.sort(key=lambda x: x._reward, reverse=True)
                for node in node_list[:self.num_beams]:
                    self.selected_node_list.append(node)

        strategy2prediction = None
        if self.output_strategy is not None:
            strategy2prediction = await self._output(final_node_list)

        output_dict = dict(all_node=[node.to_dict() for node in self.all_node_list])

        return strategy2prediction, output_dict
