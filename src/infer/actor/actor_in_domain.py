from typing import List

from src.infer.actor.base_actor import BaseActor
from src.infer.env import State, Node


class Actor(BaseActor):
    def __init__(self, *args, **kwargs):
        super(Actor, self).__init__(*args, **kwargs)

    def _get_in_domain_state(self, rec_output, node):
        rec_item_list = self.task.clean_prediction(rec_output.rec_item_list)

        rec_content_list = [self.task.get_item_info_str(item, self.task.item2info[item]) for item in rec_item_list]
        # rec_content_list = []
        # for item in rec_item_list:
        #     attr2info = dict()
        #     for k, v in self.task.item2info[item].items():
        #         if isinstance(v, list):
        #             attr2info[k] = ', '.join(v)
        #         elif isinstance(v, str):
        #             attr2info[k] = v
        #         else:
        #             raise NotImplementedError
        #
        #     rec_content = rec_content_template.format(**attr2info, item=item)
        #     rec_content_list.append(rec_content)

        complete_rec_content = ''
        for i, rec_content in enumerate(rec_content_list):
            if i > 0:
                complete_rec_content += '\n'
            complete_rec_content += f'{i + 1}. {rec_content}'

        # past_item_set = copy.copy(node.state.past_item_set)
        # if past_item_set is None:
        #     past_item_set = set()
        # if node.state.rec_item_list is not None:
        #     past_item_set |= set(node.state.rec_item_list)

        state = State(
            dialog_history_list=node.state.dialog_history_list,
            rec_item_list=rec_item_list,
            rec_content=complete_rec_content, rec_content_list=rec_content_list,
            conv_id=node.state.conv_id, turn_id=node.state.turn_id,
            user_id=node.state.user_id, user_history_list=node.state.user_history_list
        )

        return state

    async def __call__(self, node: Node) -> List[dict]:
        child_node_info_list = []

        rec_output_list = await self.rec_model(node, n=self.n)
        for rec_output in rec_output_list:
            state = self._get_in_domain_state(rec_output=rec_output, node=node)
            action = rec_output.rec_instruction
            transition = rec_output.rec_response
            child_node_info_list.append(dict(state=state, action=action, transition=transition, parent=node))

        return child_node_info_list
