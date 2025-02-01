from dataclasses import dataclass
from typing import List, Dict, Set

import numpy as np

from src.task.base_task import Dialog


@dataclass
class State(dict):
    conv_id: str
    turn_id: str
    user_id: str = None
    user_history_list: List = None
    dialog_history_list: List[Dialog] = None
    rec_item_list: List[str] = None
    past_item_set: Set[str] = None
    rec_content: str = None
    rec_content_list: List[str] = None
    rec_item_score: Dict[str, List[float]] = None

    def to_dict(self):
        return dict(
            conv_id=self.conv_id,
            turn_id=self.turn_id,
            rec_item_list=self.rec_item_list,
            rec_content=self.rec_content,
            rec_item_score=self.rec_item_score
        )


class Node:
    def __init__(self, id, state: State, action=None, transition=None, parent: "Node" = None, depth=None, reward=None):
        self.id = id
        self.state = state
        self.action = action
        self.transition = transition

        self.parent = parent
        if parent is None and depth is None:
            self.depth = 0
        elif parent is not None:
            self.depth = parent.depth + 1

            if parent.state.past_item_set is not None and parent.state.rec_item_list is not None:
                self.state.past_item_set = parent.state.past_item_set | set(parent.state.rec_item_list)
            elif parent.state.rec_item_list is not None:
                self.state.past_item_set = set(parent.state.rec_item_list)
            elif parent.state.past_item_set is not None:
                self.state.past_item_set = parent.state.past_item_set
        else:
            self.depth = depth

        self._reward = reward
        self._reward_list = None
        # self._future_reward_list = None

        self._is_terminal = False

    def to_dict(self):
        return dict(
            id=self.id,
            parent=self.parent.id if self.parent is not None else None,
            depth=self.depth,
            state=self.state.to_dict(),
            action=self.action,
            transition=self.transition,
            reward=self.reward,
            is_terminal=self.is_terminal,
            # q_value=self.q_value
        )

    @property
    def is_terminal(self):
        return self._is_terminal

    @is_terminal.setter
    def is_terminal(self, is_terminal):
        self._is_terminal = is_terminal

    @property
    def reward(self):
        if self._reward_list is None:
            return None
        else:
            self._reward = np.mean(self._reward_list)
            return self._reward

    def add_reward(self, reward: float):
        if self._reward_list is None:
            self._reward_list = []

        self._reward_list.append(reward)

    # def update_item_score(self, item_score_dict):
    #     if self.state.rec_item_score is None:
    #         self.state.rec_item_score = defaultdict(list)
    #
    #     for item, reward in item_score_dict.items():
    #         self.state.rec_item_score[item].append(reward)

    # @reward.setter
    # def reward(self, value):
    #     self._reward = value

    # def add_future_reward(self, future_reward):
    #     self._future_reward_list.append(future_reward)

    # @property
    # def q_value(self):
    #     if self.reward is not None:
    #         return self.reward + sum(self._future_reward_list)
    #     else:
    #         return None
