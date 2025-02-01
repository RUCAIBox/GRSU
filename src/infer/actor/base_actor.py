from abc import ABC, abstractmethod

from src.infer.actor.rec_model.base_rec_model import BaseRecModel
from src.infer.env import Node
from src.infer.user_model.base_user_model import BaseUserModel
from src.task.base_task import BaseTask


class BaseActor(ABC):
    def __init__(self, rec_model: BaseRecModel, user_model: BaseUserModel = None, task: BaseTask = None, n=1, init_n=None):
        self.rec_model = rec_model
        self.user_model = user_model
        self.task = task
        self.n = n
        self.init_n = init_n
        if self.init_n is None:
            self.init_n = self.n

    @abstractmethod
    async def __call__(self, node: Node):
        pass
