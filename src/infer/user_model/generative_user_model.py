import asyncio
import copy
import random
from collections import defaultdict
from typing import List, Dict

import numpy as np

from src.infer.env import State, Node
from src.infer.user_model.base_user_model import BaseUserModel
from src.model.base_model import BaseModel
from src.task.base_task import Dialog
from src.train.data.collect_instruction_data import get_response_prompt_template, item_reward_pattern


class UserModel(BaseUserModel):
    def __init__(self, *args, **kwargs):
        super(UserModel, self).__init__(*args, **kwargs)

        if self.feedback_type == 'negative':
            self.feedback_prompt_template = get_response_prompt_template('feedback')
        elif self.feedback_type == 'regular':
            self.feedback_prompt_template = get_response_prompt_template('feedback-regular')
        else:
            raise NotImplementedError

        if self.reward_prompt_type is not None:
            self.reward_prompt_template = get_response_prompt_template(self.reward_prompt_type)

        self.score_prompt_template = get_response_prompt_template(self.score_prompt_type)
        self.score_pattern = item_reward_pattern

        self.score_cache = dict()

    def _get_reward_prompt(self, state: State, prompt_template, rec_content=None):
        dialog_history_str_list = []
        for dialog_turn in state.dialog_history_list:
            dialog_turn = Dialog(**dialog_turn)
            dialog_turn_str = f'{dialog_turn.role}: {dialog_turn.text}'
            dialog_history_str_list.append(dialog_turn_str)
        dialog_history_str = '\n'.join(dialog_history_str_list)

        if rec_content is None:
            rec_content = state.rec_content

        return prompt_template.format(dialog_history=dialog_history_str, rec_content=rec_content)

    def _get_feedback_prompt_template(self):
        if self.feedback_type == 'regular':
            prompt_template = self.reward_prompt_template
        elif self.feedback_type == 'negative':
            prompt_template = '''
## Conversation

{dialog_history}

## Recommendation By System

{rec_content}

## Instruction

You are the user to look for recommendations. \
Read the "Conversation" part and provide negative feedback to the recommendations given by the system (the "Recommendation By System" part).
'''.strip()
        else:
            raise NotImplementedError

        return prompt_template

    def _get_feedback_prompt(self, state: State, rec_content=None):
        dialog_history_str_list = []
        for dialog_turn in state.dialog_history_list:
            dialog_turn = Dialog(**dialog_turn)
            dialog_turn_str = f'{dialog_turn.role}: {dialog_turn.text}'
            dialog_history_str_list.append(dialog_turn_str)
        dialog_history_str = '\n'.join(dialog_history_str_list)

        if rec_content is None:
            rec_content = state.rec_content

        return self.feedback_prompt_template.format(dialog_history=dialog_history_str, rec_content=rec_content)

    async def provide_feedback(self, state: State, n=1) -> List[str]:
        feedback_list = []

        prompt = self._get_feedback_prompt(state)

        sampling_params = copy.copy(self.feedback_sampling_params)

        count = 0
        while len(feedback_list) != n:
            count += 1
            if count > 3 and len(feedback_list) > 0:
                break
            # if count > 3:
            #     break
            if count > 1 and sampling_params['temperature'] == 0:
                sampling_params['temperature'] = 1
                if 'seed' in sampling_params:
                    sampling_params.pop('seed')

            cur_n = n - len(feedback_list)

            response_list = await self.feedback_model.generate(prompt, n=cur_n, **sampling_params)
            if response_list is None:
                break
            for response in response_list:
                if self.feedback_type == 'score':
                    response = response.message['content']
                    item_list = []
                    for line in response.split('\n'):
                        match = self.score_pattern.search(line)
                        if match is None:
                            continue
                        item_list.append(match.group(1))

                    if len(item_list) == len(state.rec_item_list):
                        feedback_list.append(response)
                        self.log(f'[provide feedback]\nprompt: {prompt}\nvalid response: {response}')
                    else:
                        self.log(f'[provide feedback]\nprompt: {prompt}\ninvalid response: {response}')
                elif self.feedback_type in {'negative', 'regular'}:
                    if response.finish_reason == 'length':
                        self.log(f'[provide feedback]\nprompt: {prompt}\ninvalid response: {response.message["content"]}')
                        continue
                    elif response.finish_reason == 'stop':
                        feedback_list.append(response.message['content'])
                        self.log(f'[provide feedback]\nprompt: {prompt}\nvalid response: {response.message["content"]}')
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError

        return feedback_list

    async def score_item(
            self, state: State, is_reward, rec_item_list: List[str] = None, rec_content: str = None,
            model: BaseModel = None, sampling_params: dict = None, prompt_template: str = None
    ) -> Dict[str, List[float]]:
        item_score_dict = defaultdict(list)

        rec_item_list = state.rec_item_list if rec_item_list is None else rec_item_list

        model = model if model is not None else self.score_model
        if sampling_params is None:
            sampling_params = copy.copy(self.score_sampling_params)
            sampling_params['logprobs'] = True

        prompt_template = prompt_template
        if prompt_template is None:
            prompt_template = self.score_prompt_template

        use_cache = self.score_type == 'no-candidate' and sampling_params['temperature'] == 0

        async def _score_item(rec_content, rec_item_list):
            if use_cache:
                all_has_cache = True
                for item in rec_item_list:
                    if item not in self.score_cache:
                        all_has_cache = False
                        break
                if all_has_cache is True:
                    cur_item_score_dict = defaultdict(list)
                    for item in rec_item_list:
                        score = self.score_cache[item]
                        item_score_dict[item].append(score)
                        cur_item_score_dict[item].append(score)
                    self.log(f'[score item]\nhit cache\nscore: {dict(cur_item_score_dict)}')
                    return

            prompt = self._get_reward_prompt(state=state, prompt_template=prompt_template, rec_content=rec_content)

            count = 0
            flag = False
            while count < 5 and flag is False:
                count += 1

                if count > 1 and sampling_params['temperature'] == 0:
                    sampling_params['temperature'] = 1
                    if 'seed' in sampling_params:
                        sampling_params.pop('seed')

                response_list = await model.generate(prompt, **sampling_params)
                if response_list is None:
                    break
                for response in response_list:
                    item_name_list = []
                    for line in response.message["content"].split('\n'):
                        match = self.score_pattern.search(line)
                        if match is None:
                            continue
                        item_name = match.group(1)
                        item_name_list.append(item_name)

                    if set(rec_item_list) != set(item_name_list):
                        self.log(
                            f'[score item]\nprompt: {prompt}\ninvalid response: {response.message["content"]}'
                        )
                        continue

                    score_list = []
                    logprobs = response.logprobs["content"]
                    for i in range(len(logprobs) - 1):
                        if logprobs[i]['token'].strip() == ')?':
                            if logprobs[i + 1]['token'].strip() == 'Yes':
                                score = np.exp(logprobs[i + 1]['logprob'])
                                score_list.append(score)
                            elif logprobs[i + 1]['token'].strip() == 'No':
                                score = 1 - np.exp(logprobs[i + 1]['logprob'])
                                score_list.append(score)

                    if len(score_list) != len(item_name_list):
                        self.log(
                            f'[score item]\nprompt: {prompt}\ninvalid response: {response.message["content"]}'
                        )
                        continue

                    cur_item_score_dict = defaultdict(list)
                    for item_name, score in zip(item_name_list, score_list):
                        item_score_dict[item_name].append(score)
                        cur_item_score_dict[item_name].append(score)
                        if use_cache:
                            self.score_cache[item_name] = score

                    self.log(
                        f'[score item]\nprompt: {prompt}\nvalid response: {response.message["content"]}\nscore: {dict(cur_item_score_dict)}'
                    )

                    flag = True

        if self.score_type == 'no-candidate' and is_reward is True:
            task_list = []
            for rec_content, rec_item in zip(state.rec_content_list, state.rec_item_list):
                task_list.append(_score_item(rec_content=f'1. {rec_content}', rec_item_list=[rec_item]))
            await asyncio.gather(*task_list)
        else:
            await _score_item(rec_content=rec_content, rec_item_list=rec_item_list)

        return item_score_dict

    def _diversity_reward(self, state: State):
        cur_item_set = set(state.rec_item_list)
        overlap_item_num = 0 if state.past_item_set is None else len(cur_item_set & state.past_item_set)
        reward = (len(cur_item_set) - overlap_item_num) / len(state.rec_item_list)
        return reward

    async def _relevance_reward(self, state: State):
        item_score_dict = await self.score_item(
            state=state, model=self.reward_model, sampling_params=self.reward_sampling_params, is_reward=True
        )

        if state.rec_item_score is None:
            state.rec_item_score = defaultdict(list)
        avg_score_list = []
        for item, score_list in item_score_dict.items():
            state.rec_item_score[item].extend(score_list)
            avg_score_list.extend(score_list)

        score = 0
        if len(avg_score_list) > 0:
            if 'max' in self.reward_type:
                score = np.max(avg_score_list)
            else:
                score = np.mean(avg_score_list)
        return score

    async def evaluate_reward(self, node: Node) -> float:
        if self.reward_type == 'diversity':
            reward = self._diversity_reward(node.state)
        elif self.reward_type in ('relevance', 'relevance-max'):
            reward = await self._relevance_reward(node.state)
        elif self.reward_type in ('diversity+relevance', 'relevance+diversity'):
            reward = self._diversity_reward(node.state) + await self._relevance_reward(node.state)
        elif self.reward_type == 'random':
            reward = random.random()
        else:
            raise NotImplementedError

        self.log(f'evaluate reward\nnode id: {node.id}, reward: {reward}')

        return reward

    # async def evaluate_reward(self, node: Node):
        # sampling_params = copy.copy(self.reward_sampling_params)
        # sampling_params['logprobs'] = True

        # if self.reward_type == 'single':
        #     for rec_item, rec_content in zip(node.state.rec_item_list, node.state.rec_content_list):
        #         item_key = (node.state.conv_id, node.state.turn_id, rec_item)
        #         if item_key in self.item2reward:
        #             item_score_dict = {rec_item: self.item2reward[item_key]}
        #             node.update_item_score(item_score_dict)
        #             continue
        #
        #         prompt = self._get_reward_prompt(node.state, rec_content)
        #
        #         flag = False
        #         count = 0
        #         while count < 3 and flag is False:
        #             count += 1
        #
        #             if count > 1 and sampling_params['temperature'] == 0:
        #                 sampling_params['temperature'] = 1
        #                 if 'seed' in sampling_params:
        #                     sampling_params.pop('seed')
        #
        #             response_list = await self.reward_model.generate(prompt, **sampling_params)
        #             for response in response_list:
        #                 item_name = None
        #                 for line in response.message["content"].split('\n'):
        #                     match = self.item_reward_pattern.search(line)
        #                     if match is None:
        #                         continue
        #                     item_name = match.group(1)
        #
        #                 if item_name is None:
        #                     self.log(
        #                         f'evaluate reward\nprompt: {prompt}\ninvalid response: {response.message["content"]}'
        #                     )
        #                     continue
        #
        #                 score = None
        #                 logprobs = response.logprobs['content']
        #                 for i in range(len(logprobs) - 1):
        #                     if logprobs[i]['token'].strip() == ')?':
        #                         if logprobs[i + 1]['token'].strip() == 'Yes':
        #                             score = np.exp(logprobs[i + 1]['logprob'])
        #                         elif logprobs[i + 1]['token'].strip() == 'No':
        #                             score = 1 - np.exp(logprobs[i + 1]['logprob'])
        #
        #                 if score is None:
        #                     self.log(
        #                         f'evaluate reward\nprompt: {prompt}\ninvalid response: {response.message["content"]}'
        #                     )
        #                     continue
        #
        #                 item_score_dict = {item_name: score}
        #
        #                 self.item2reward[(node.state.conv_id, node.state.turn_id, item_name)] = score
        #
        #                 node.update_item_score(item_score_dict)
        #                 flag = True
        #                 self.log(
        #                     f'evaluate reward\nprompt: {prompt}\nvalid response: {response.message["content"]}\nscore: {item_score_dict}'
        #                 )
        # else:
        #     prompt = self._get_reward_prompt(node.state)
        #
        #     flag = False
        #     count = 0
        #     while count < 3 and flag is False:
        #         count += 1
        #
        #         if count > 1 and sampling_params['temperature'] == 0:
        #             sampling_params['temperature'] = 1
        #             if 'seed' in sampling_params:
        #                 sampling_params.pop('seed')
        #
        #         response_list = await self.reward_model.generate(prompt, **sampling_params)
        #         for response in response_list:
        #             if self.reward_type == 'list':
        #                 reward = None
        #
        #                 logprobs = response.logprobs["content"][-3:]
        #                 for logprob in logprobs[::-1]:
        #                     token = logprob.token.strip()
        #                     token_logprob = logprob.logprob
        #                     if token == 'Yes':
        #                         reward = np.exp(token_logprob)
        #                     elif token == 'No':
        #                         reward = 1 - np.exp(token_logprob)
        #
        #                     if reward is not None:
        #                         break
        #
        #                 if reward is not None:
        #                     node.update_reward([reward])
        #                     flag = True
        #                     self.log(f'evaluate reward\nprompt: {prompt}\nvalid response: {response.message["content"]}\ntoken: {"|".join([logprob.token for logprob in logprobs])}\nscore: {reward}')
        #                 else:
        #                     self.log(f'evaluate reward\nprompt: {prompt}\ninvalid response: {response.message["content"]}\ntoken: {"|".join([logprob.token for logprob in logprobs])}')
        #
        #             elif self.reward_type == 'item':
        #                 item_name_list = []
        #                 for line in response.message["content"].split('\n'):
        #                     match = self.item_reward_pattern.search(line)
        #                     if match is None:
        #                         continue
        #                     item_name = match.group(1)
        #                     item_name_list.append(item_name)
        #
        #                 if len(item_name_list) != len(node.state.rec_item_list):
        #                     self.log(
        #                         f'evaluate reward\nprompt: {prompt}\ninvalid response: {response.message["content"]}'
        #                     )
        #                     continue
        #
        #                 score_list = []
        #                 logprobs = response.logprobs["content"]
        #                 for i in range(len(logprobs) - 1):
        #                     if logprobs[i]['token'].strip() == ')?':
        #                         if logprobs[i + 1]['token'].strip() == 'Yes':
        #                             score = np.exp(logprobs[i + 1]['logprob'])
        #                             score_list.append(score)
        #                         elif logprobs[i + 1]['token'].strip() == 'No':
        #                             score = 1 - np.exp(logprobs[i + 1]['logprob'])
        #                             score_list.append(score)
        #
        #                 if len(score_list) != len(item_name_list):
        #                     self.log(
        #                         f'evaluate reward\nprompt: {prompt}\ninvalid response: {response.message["content"]}'
        #                     )
        #                     continue
        #
        #                 item_score_dict = dict()
        #                 for item_name, score in zip(item_name_list, score_list):
        #                     item_score_dict[item_name] = score
        #
        #                 node.update_item_score(item_score_dict)
        #                 flag = True
        #                 self.log(
        #                     f'evaluate reward\nprompt: {prompt}\nvalid response: {response.message["content"]}\nscore: {item_score_dict}'
        #                 )
        #
        #                 # item_key = (node.state.conv_id, node.state.turn_id, tuple(item_name_list))
        #                 # if item_key not in self.item2reward:
        #                 #     self.item2reward[item_key] = [response]
        #                 # else:
        #                 #     self.item2reward[item_key].append(response)
        #
        #             else:
        #                 raise NotImplementedError
