from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from src.infer.env import Node


@dataclass
class RecModelOutput:
    rec_item_list: List[str]
    rec_instruction: str
    rec_response: str


class BaseRecModel(ABC):
    def __init__(self, task, k, log_file_path=None, *args, **kwargs):
        self.task = task
        self.k = k
        self.log_file_path = log_file_path

    @abstractmethod
    async def __call__(self, node: Node, n=1, *args, **kwargs):
        pass

    def _log(self, log_str):
        if self.log_file_path is not None:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(log_str + '\n\n')
                # await f.flush()
