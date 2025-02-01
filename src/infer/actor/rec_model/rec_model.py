import copy
import re
from typing import List, Dict

from src.infer.actor.rec_model.base_rec_model import BaseRecModel, RecModelOutput
from src.infer.env import State, Node
from src.model.base_model import BaseModel
from src.task.base_task import Dialog


class RecModel(BaseRecModel):
    def __init__(
            self,
            model_dict: Dict[str, BaseModel], model_name: str, sampling_params: dict,
            allow_user=False, different_from_past=True, no_year=False,
            *args, **kwargs
    ):
        super(RecModel, self).__init__(*args, **kwargs)

        self.model = model_dict[model_name]
        self.sampling_params = sampling_params

        # self.item_line_pattern = re.compile(r'^\d+\.\s')
        self.item_name_pattern = re.compile(r'\*\*(.*?)\*\*')
        # self.item_name_pattern = re.compile(r'\d+\. (.*)')

        self.prompt_template = self._get_prompt_template(no_year=no_year)
        self.prompt_with_feedback_template = self._get_prompt_with_feedback_template(
            allow_user=allow_user, different_from_past=different_from_past, no_year=no_year
        )

    def _get_prompt_template(self, no_year):
        if no_year is True:
            format_instruction = \
"""
Each line of your response should follow this format: [index]. **[item name]**. \
NOTE: No year and description is needed for the item to recommend in each line of your response.
""".strip()
        else:
            format_instruction = \
"""
Each line of your response should follow this format: [index]. **[item name] (year)**. \
NOTE: No description is needed for the item to recommend in each line of your response.
""".strip()

        prompt_template = \
"""
## Conversation

{dialog_history}

## Instruction

You are the system to make recommendations for users. Read the "Conversation" part and recommend {k} items. \
If there is no useful information in the conversation for recommendation, randomly recommend {k} items.
""".strip()
        prompt_template += f' {format_instruction}'

        return prompt_template

    def _get_prompt_with_feedback_template(self, allow_user, different_from_past, no_year):
        rec_instruction = 'Read the "Conversation" part, revise your recommendation according to the feedback from the user (the "Feedback" part), and recommend {k} items'
        if different_from_past is True:
            rec_instruction += ' different from existing ones in the "Recommendation" part'
        rec_instruction += '.'
        if allow_user is True:
            rec_instruction += ' You can recommend those mentioned by the user if you think they are suitable.'

        if no_year is True:
            format_instruction = 'Each line of your response should follow this format: [index]. **[item name]** (no year).'
        else:
            format_instruction = 'Each line of your response should follow this format: [index]. **[item name] (year)**.'
        format_instruction += ' NOTE: No description is needed for the item to recommend in each line of your response.'

        prompt_template = \
"""
## Conversation

{dialog_history}

## Recommendation

System: {rec_content}

## Feedback

User: {feedback}

## Instruction

You are the system to make recommendations for users.
""".strip()
        prompt_template += f' {rec_instruction} {format_instruction}'
        return prompt_template

    def _get_action_prompt(self, state: State, *args, **kwargs):
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
            prompt = self.prompt_with_feedback_template.format(dialog_history=dialog_history_str, rec_content=state.rec_content, feedback=feedback, k=self.k)
        else:
            prompt = self.prompt_template.format(dialog_history=dialog_history_str, k=self.k)
        return prompt

    def _item_extraction(self, response):
        response_line_list = [line for line in response.split('\n') if self.item_name_pattern.search(line) is not None]
        item_list = []
        for line in response_line_list:
            item_str = self.item_name_pattern.search(line)
            item_str = item_str.group(1)
            item_list.append(item_str)
        return item_list

    async def __call__(self, node: Node, n=1, *args, **kwargs) -> List[RecModelOutput]:
        rec_output_list = []
        prompt = self._get_action_prompt(node.state, *args, **kwargs)

        count = 0
        while len(rec_output_list) != n:
            count += 1
            if count > 5 and len(rec_output_list) > 0:
                break

            sampling_params = copy.copy(self.sampling_params)
            if count > 1 and self.sampling_params['temperature'] == 0:
                sampling_params['temperature'] = 1
                if 'seed' in sampling_params:
                    sampling_params.pop('seed')
            cur_n = n - len(rec_output_list)

            response_list = await self.model.generate(prompt=prompt, n=cur_n, **sampling_params)
            if response_list is None:
                break

            for response in response_list:
                response = response.message.content
                rec_item_list = self._item_extraction(response)
                if len(rec_item_list) != self.k:
                    self._log(f'[recommend]\nprompt: {prompt}\ninvalid response: {response}')
                    continue
                else:
                    self._log(f'[recommend]\nprompt: {prompt}\nvalid response: {response}\nrec item list: {rec_item_list}')

                    response = response.lstrip('System:').lstrip()
                    rec_output = RecModelOutput(rec_item_list=rec_item_list, rec_instruction=prompt, rec_response=response)
                    rec_output_list.append(rec_output)

        return rec_output_list
