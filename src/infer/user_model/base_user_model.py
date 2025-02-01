from abc import ABC, abstractmethod
from typing import List, Dict

import aiofiles

from src.infer.env import State, Node
from src.model.base_model import BaseModel


class BaseUserModel(ABC):
    feedback_type = {'regular', 'negative'}
    reward_type = {'relevance', 'relevance-max', 'random'}
    score_type = {'no-candidate', 'candidate-aware'}

    def __init__(self, k, model_dict: Dict[str, BaseModel], feedback_params: dict = None, reward_params: dict = None, score_params: dict = None, log_file_path=None):
        self.k = k
        # feedback param
        # model_name = feedback_params.pop('model_name')
        if feedback_params is not None:
            self.feedback_type = feedback_params['type']
            assert self.feedback_type in BaseUserModel.feedback_type

            self.feedback_model = model_dict[feedback_params['model_name']]
            self.feedback_sampling_params = feedback_params['sampling_params']

        # reward param
        # model_name = reward_params.pop('model_name')
        self.reward_prompt_type = None
        if reward_params is not None:
            self.reward_type = reward_params['type']
            assert self.reward_type in BaseUserModel.reward_type

            if 'relevance' in self.reward_type:
                self.reward_model = model_dict[reward_params['model_name']]
                self.reward_sampling_params = reward_params['sampling_params']
                if 'prompt_type' in reward_params:
                    self.reward_prompt_type = reward_params['prompt_type']
                else:
                    self.reward_prompt_type = 'reward'

        # score param
        if score_params is not None:
            if 'type' in score_params:
                self.score_type = score_params['type']
            else:
                self.score_type = 'candidate-aware'
            assert self.score_type in BaseUserModel.score_type

            self.score_model = model_dict[score_params['model_name']]
            self.score_sampling_params = score_params['sampling_params']

            if 'prompt_type' in score_params:
                self.score_prompt_type = score_params['prompt_type']
            else:
                self.score_prompt_type = 'reward'

        self.log_file_path = log_file_path

    def log(self, log_str):
        if self.log_file_path is not None:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(log_str + '\n\n')
                # await f.flush()

    @abstractmethod
    async def provide_feedback(self, state: State, n=1) -> List[str]:
        pass

    @abstractmethod
    async def evaluate_reward(self, node: Node) -> float:
        pass

    @abstractmethod
    async def score_item(self, state: State, is_reward, rec_item_list: List[str] = None, rec_content: str = None, model: BaseModel = None, sampling_params: dict = None) -> Dict[str, List[float]]:
        pass
