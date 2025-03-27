import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Union, Any, Optional, List


class TextDataset(Dataset):

    def __init__(
            self,
            dataset,
            data_args,
            split,  # 'train', 'test', 'val'
    ):
        super().__init__()
        self.dataset = dataset
        self.length = len(self.dataset[split])
        self.data_args = data_args
        self.split = split

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = {
            'mask': np.array(self.dataset[self.split][idx]['mask']),
            'sn_sp_repr': np.array(self.dataset[self.split][idx]['sn_sp_repr']),
            'sn_input_ids': np.array(self.dataset[self.split][idx]['sn_input_ids']),
            'indices_pos_enc': np.array(self.dataset[self.split][idx]['indices_pos_enc']),
            'sn_repr_len': np.array(self.dataset[self.split][idx]['sn_repr_len']),
            'words_for_mapping': self.dataset[self.split][idx]['words_for_mapping'],
        }
        return sample


def text_dataset_loader(
        data,
        data_args,
        split: str,
        deterministic: bool = False,
):
    dataset = TextDataset(
        dataset=data,
        data_args=data_args,
        split=split,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=data_args.batch_size,
        shuffle=not deterministic,
        num_workers=0,
    )
    return iter(data_loader)


class FixdurDataset(Dataset):
    def __init__(
        self,
        data: Dict[str, Union[torch.Tensor, Any]],
    ):
        super().__init__()
        self.data = data
    
    def __len__(self):
        return len(self.data['sp_embeddings'])
    
    def __getitem__(self, idx):
        sample = {
            'sp_embeddings': self.data['sp_embeddings'][idx],
            'attention_mask': self.data['attention_mask'][idx],
            'unique_idx': self.data['unique_idx'][idx],
        }
        return sample