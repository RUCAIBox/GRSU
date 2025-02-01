from typing import List

from src.infer.actor.base_actor import BaseActor
from src.infer.env import State, Node


class Actor(BaseActor):
    def __init__(self, *args, **kwargs):
        super(Actor, self).__init__(*args, **kwargs)

    async def __call__(self, node: Node) -> List[dict]:
        child_node_info_list = []

        rec_output_list = await self.rec_model(node, n=self.n)
        for rec_output in rec_output_list:
            # past_item_set = copy.copy(node.state.past_item_set)
            # if past_item_set is None:
            #     past_item_set = set()
            # past_item_set |= set(node.state.rec_item_list)

            state = State(
                dialog_history_list=node.state.dialog_history_list,
                rec_item_list=rec_output.rec_item_list,
                rec_content=rec_output.rec_response,
                conv_id=node.state.conv_id, turn_id=node.state.turn_id
            )
            action = rec_output.rec_instruction
            transition = rec_output.rec_response
            child_node_info_list.append(dict(state=state, action=action, transition=transition, parent=node))

        return child_node_info_list
