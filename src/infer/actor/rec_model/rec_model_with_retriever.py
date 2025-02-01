import copy

import torch
import torch.nn.functional as F

from typing import List, Dict

from src.infer.actor.rec_model.base_rec_model import BaseRecModel, RecModelOutput
from src.infer.env import State, Node
from src.model.base_model import BaseModel
from src.task.base_task import Dialog


class RecModel(BaseRecModel):
    prefix_tuple = ('Query:', '**Query:**', '**Query**:', 'Query\n')

    def __init__(self, model_dict: Dict[str, BaseModel], query_params: dict, retrieve_params: dict, *args, **kwargs):
        super(RecModel, self).__init__(*args, **kwargs)

        self.query_model = model_dict[query_params['model_name']]
        self.query_sampling_params = query_params['sampling_params']
        self.prompt_template = self._get_prompt_template()
        self.prompt_with_feedback_template = self._get_prompt_with_feedback_template()

        self.retrieve_model = model_dict[retrieve_params['model_name']]
        self.task_description = 'Given a query, retrieve relevant entity descriptions'

    def _get_prompt_template(self):
        prompt_template = \
"""
## Conversation

{dialog_history}

## Instruction

You are the system to make recommendations for users. \
Read the "Conversation" part and generate a textual query for another retrieval model to retrieve items for recommendation. \
In your response, you should first analyze the user preference and then generate the textual query. \
NOTE: No item name should occur in the query. \
Your response should follow this format:
Analysis: [analysis]
Query: [query]
""".strip()
        return prompt_template

    def _get_prompt_with_feedback_template(self):
        prompt_template = \
"""
## Conversation

{dialog_history}

## Recommendation

System: {rec_content}

## Feedback

User: {feedback}

## Instruction

You are the system to make recommendations for users. \
Read the "Conversation", "Recommendation", and "Feedback" part and generate a textual query for another retrieval model to retrieve items for recommendation. \
In your response, you should first analyze the user preference and then generate the textual query. \
NOTE: No item name should occur in the query. \
Your response should follow this format:
Analysis: [analysis]
Query: [query]
""".strip()
        return prompt_template

    def _get_query_prompt(self, state: State, *args, **kwargs):
        feedback = kwargs.get('feedback', None)
        if feedback is None and len(args) == 1:
            feedback = args[0]

        dialog_history_str_list = []
        for dialog_turn in state.dialog_history_list:
            dialog_turn = Dialog(**dialog_turn)
            dialog_turn_str = f'{dialog_turn.role}: {dialog_turn.text}'
            dialog_history_str_list.append(dialog_turn_str)
        dialog_history_str = '\n'.join(dialog_history_str_list)

        if feedback is not None:
            prompt = self.prompt_with_feedback_template.format(dialog_history=dialog_history_str, rec_content=state.rec_content, feedback=feedback)
        else:
            prompt = self.prompt_template.format(dialog_history=dialog_history_str)
        return prompt

    def _get_retrieve_prompt(self, query):
        return f'Instruct: {self.task_description}\nQuery: {query}'

    async def __call__(self, node: Node, n=1, *args, **kwargs) -> List[RecModelOutput]:
        rec_output_list = []
        prompt = self._get_query_prompt(node.state, *args, **kwargs)

        all_response_list = []
        query_list = []
        count = 0
        extend_length = False
        while len(query_list) != n:
            if count > 3 and len(query_list) > 0:
                break

            sampling_params = copy.copy(self.query_sampling_params)
            if count > 1 and sampling_params['temperature'] == 0:
                sampling_params['temperature'] = 1
                if 'seed' in sampling_params:
                    sampling_params.pop('seed')
            if extend_length is True:
                sampling_params['max_tokens'] = self.query_sampling_params['max_tokens'] * 2
            cur_n = n - len(query_list)

            response_list = await self.query_model.generate(prompt=prompt, n=cur_n, **sampling_params)
            for response in response_list:
                flag = False
                if response.finish_reason == 'stop':
                    content = response.message.content
                    for prefix in RecModel.prefix_tuple:
                        if prefix in content:
                            query = content.split(prefix)[-1]
                            query_list.append(query)
                            all_response_list.append(content)
                            flag = True
                            break
                elif response.finish_reason == 'length':
                    extend_length = True
                else:
                    print('recommend', response.finish_reason)
                    raise NotImplementedError
                if flag is False:
                    self._log(f'recommend\nprompt: {prompt}\ninvalid response: {response.message.content}')

        retrieve_input_list = [self._get_retrieve_prompt(query) for query in query_list]
        embedding_list = await self.retrieve_model.get_embedding(retrieve_input_list)

        for response, embedding in zip(all_response_list, embedding_list):
            embedding = F.normalize(torch.tensor(embedding).unsqueeze(0), p=2, dim=-1)
            score = embedding @ self.task.item_embeddings.T  # (1, n_item)

            # indices = score.squeeze(0).topk(k=self.k).indices.tolist()
            # rec_item_list = [self.task.id2item[idx] for idx in indices]

            if node.state.past_item_set is None:
                indices = score.squeeze(0).topk(k=self.k).indices.tolist()
                rec_item_list = [self.task.id2item[idx] for idx in indices]
            else:
                indices = score.squeeze(0).argsort(descending=True).tolist()
                rec_item_list = []
                for index in indices:
                    item = self.task.id2item[index]
                    if item not in node.state.past_item_set:
                        rec_item_list.append(item)
                        if len(rec_item_list) == self.k:
                            break

            self._log(f'recommend\nprompt: {prompt}\nvalid response: {response}\nrec item list: {rec_item_list}')

            response = response.lstrip('System:').lstrip()
            rec_output = RecModelOutput(rec_item_list=rec_item_list, rec_instruction=prompt, rec_response=response)
            rec_output_list.append(rec_output)

        return rec_output_list
