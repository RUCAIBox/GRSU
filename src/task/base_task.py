import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Set, Dict, Any

import numpy as np
from datasets import Dataset
from rapidfuzz import process, fuzz, utils
from torch.utils.data import DataLoader


@dataclass
class Dialog(dict):
    role: str
    text: str

    def __post_init__(self):
        # 将 dataclass 属性添加到字典中
        self.update(self.__dict__)


@dataclass
class TaskData(dict):
    conv_id: str
    turn_id: str
    dialog_history_list: List[Dialog]
    rec_label_list: List[str]
    user_id: str = None
    user_history_list: List = None

    def __post_init__(self):
        # 将 dataclass 属性添加到字典中
        self.update(self.__dict__)


fuzzy_scorer_dict = {
    'ratio': fuzz.ratio,
    'partial-ratio': fuzz.partial_ratio,
    'WRatio': fuzz.WRatio,
    'QRatio': fuzz.QRatio,
}


class BaseTask(ABC):
    def __init__(self, data_file_path, item_file_path, only_new=False, data_size=None, k_list=None, prediction_need_clean=True, fuzzy_scorer='QRatio'):
        self.item2info = self.load_candidate_item_info(item_file_path=item_file_path)
        self.item_set = set(self.item2info.keys())
        self.dataset = self._load_dataset(data_file_path=data_file_path, data_size=data_size, only_new=only_new)

        self.prediction_need_clean = prediction_need_clean
        self.cache_for_clean_prediction = dict()
        self.fuzzy_scorer = fuzzy_scorer_dict[fuzzy_scorer]

        self.k_list = k_list
        if self.k_list is None:
            self.k_list = [1, 5, 10, 20, 50]

        self.id2item = {idx: item for idx, item in enumerate(self.item_set)}
        self.item_embeddings = None

    def set_item_embeddings(self, item_embeddings):
        self.item_embeddings = item_embeddings

    @abstractmethod
    def _load_dataset(self, data_file_path, only_new, data_size=None) -> Dataset:
        pass

    @classmethod
    def get_item_info_str(cls, item, info_dict):
        pass

    def get_item_info_str_for_embedding(self, item):
        pass

    @classmethod
    def load_candidate_item_info(cls, item_file_path) -> Dict[str, Dict[str, Any]]:
        pass

    def get_dataloader(self, batch_size=1, shuffle=False):
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: x)

    def clean_prediction(self, pred_item_list):
        cleaned_pred_item_list = []
        for item in pred_item_list:
            if item in self.cache_for_clean_prediction:
                cleaned_item = self.cache_for_clean_prediction[item]
            else:
                if item in self.item_set:
                    cleaned_item = item
                else:
                    cleaned_item = process.extractOne(
                        item, self.item_set, scorer=self.fuzzy_scorer, processor=utils.default_process
                    )[0]

            cleaned_pred_item_set = set(cleaned_pred_item_list)
            if cleaned_item in cleaned_pred_item_set:
                item_set = self.item_set - cleaned_pred_item_set
                cleaned_item = process.extractOne(
                    item, item_set, scorer=self.fuzzy_scorer, processor=utils.default_process
                )[0]
            else:
                self.cache_for_clean_prediction[item] = cleaned_item

            cleaned_pred_item_list.append(cleaned_item)

        return cleaned_pred_item_list

    def _cal_hit(self, pred_item_list, label_item):
        return 1 if label_item in pred_item_list else 0

    def _cal_ndcg(self, pred_item_list, label_item):
        ndcg = 0
        if label_item in pred_item_list:
            label_rank = pred_item_list.index(label_item)
            ndcg = 1 / math.log2(label_rank + 2)
        return ndcg

    def _cal_mrr(self, pred_item_list, label_item):
        mrr = 0
        if label_item in pred_item_list:
            label_rank = pred_item_list.index(label_item)
            mrr = 1 / (label_rank + 1)
        return mrr

    def cal_metric(self, pred_item_list, label_item_list):
        metric_dict = defaultdict(list)

        if self.prediction_need_clean is True:
            pred_item_list = self.clean_prediction(pred_item_list)

        for k in self.k_list:
            for label_item in label_item_list:
                pred_k_list = pred_item_list[:k]
                hit = self._cal_hit(pred_k_list, label_item)
                metric_dict[f'Recall@{k}'].append(hit)

        for k in self.k_list:
            for label_item in label_item_list:
                ndcg = self._cal_ndcg(pred_k_list, label_item)
                metric_dict[f'NDCG@{k}'].append(ndcg)

        for k in self.k_list:
            for label_item in label_item_list:
                mrr = self._cal_mrr(pred_k_list, label_item)
                metric_dict[f'MRR@{k}'].append(mrr)

        for k, v in metric_dict.items():
            metric_dict[k] = np.mean(v)

        return pred_item_list, metric_dict
