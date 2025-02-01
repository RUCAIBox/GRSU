import copy
import json
from typing import Dict, Any

from datasets import Dataset

from src.task.base_task import BaseTask, TaskData, Dialog


class Task(BaseTask):
    attr_list = ['genre', 'star', 'director', 'writer', 'plot']

    def _load_dataset(self, data_file_path, only_new, data_size=None) -> Dataset:
        data_list = []
        with open(data_file_path, encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                conv_id = data['conversationId']
                dialog_history_list = []
                item_set = set()

                for message in data['messages']:
                    rec_label_list = message['item']
                    rec_label_list = [item for item in rec_label_list if item in self.item_set]
                    rec_label_set = set(rec_label_list)

                    if message['role'] == 'system' and len(rec_label_list) > 0 and len(dialog_history_list) > 0:
                        if only_new is True and len(rec_label_set & item_set) > 0:
                            continue
                        data_list.append(TaskData(
                            conv_id=conv_id, turn_id=message['turn_id'],
                            dialog_history_list=copy.copy(dialog_history_list),
                            rec_label_list=rec_label_list,
                            # response=message['text']
                        ))

                    dialog_history_list.append(Dialog(
                        role=message['role'].title(),
                        text=message['text']
                    ))
                    item_set |= rec_label_set

        data_list = data_list[:data_size]
        dataset = Dataset.from_list(data_list)
        return dataset

    @classmethod
    def get_item_info_str(cls, item, info_dict):
        info_str_list = [f'{item}']
        for attr in Task.attr_list:
            if attr in info_dict:
                value = info_dict[attr]
                if isinstance(value, list):
                    value_str = ', '.join(value)
                elif isinstance(value, str):
                    value_str = value
                else:
                    raise NotImplementedError
                if len(value_str) > 0:
                    info_str_list.append(f'  - {attr}: {value_str}')
        info_attr = '\n'.join(info_str_list)
        return info_attr

    def get_item_info_str_for_embedding(self, item):
        item_info_str_list = []
        for attr in Task.attr_list:
            if attr in self.item2info[item].items():
                value = self.item2info[item][attr]
                if isinstance(value, list):
                    value_str = ', '.join(value)
                elif isinstance(value, str):
                    value_str = value
                else:
                    raise NotImplementedError
                if len(value_str) > 0:
                    item_info_str_list.append(f'{attr}: {value_str}')
        item_info_str = f'{item}: {"; ".join(item_info_str_list)}'
        item_info_str = item_info_str.rstrip('.') + '.'
        return item_info_str

    @classmethod
    def load_candidate_item_info(cls, item_file_path) -> Dict[str, Dict[str, Any]]:
        # item_set = set()
        # with open(item_file_path, encoding='utf-8', newline='') as f:
        #     csv_reader = csv.DictReader(f)
        #
        #     for row in csv_reader:
        #         item = row['movieName']
        #         item = re.sub(r'\s+', ' ', item).strip()
        #         item_set.add(item)
        #
        # return item_set

        with open(item_file_path, encoding='utf-8') as f:
            item2info = json.load(f)
        return item2info
