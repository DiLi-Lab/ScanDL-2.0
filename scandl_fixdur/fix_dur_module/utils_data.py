"""
Utils for the fixation duration module. 
"""

import torch
from datasets import load_from_disk
from typing import Dict, Any, List, Optional, Union
import transformers
from torch.utils.data import Dataset
import datasets 
from tqdm import tqdm
import random


# def get_input_embeddings(
#     data_instance: Dict[str, Any],
#     tokenizer: transformers.GPT2TokenizerFast,
#     gpt2_model: transformers.GPT2Model,
#     aggregate: str = 'mean', # 'mean', 'sum'
#     #max_length: int = 128,
# ):
#     # dummy code for now
#     sn_repr_len = data_instance['sn_repr_len']

#     # the sentence 
#     sn_words = data_instance['words_for_mapping'].split()
#     while sn_words[-1] == '[PAD]':
#         sn_words.pop()

#     # the scanpath 
#     # remove the CLS token (already have a SEP token at the end of the sentence and the two will beconcatenated)
#     # the SEP token can stay
#     sp_ids = data_instance['sn_sp_repr'][sn_repr_len:][1:]
#     # cut off the trialing pad tokens
#     while sp_ids[-1] == 127:
#         sp_ids.pop()
#     # get the scanpath as fixated words 
#     sp_words = list()
#     for sp_id in sp_ids:
#         sp_words.append(sn_words[sp_id])
    
#     # TODO doesn't make sense to have CLS sn SEP sp SEP for auto-regressive model. BOS token
#     # join sentence and scanpath into strings and concatenate them
#     sn = ' '.join(sn_words)
#     sp = ' '.join(sp_words)
#     sn_sp = sn + ' ' + sp

#     encoded = tokenizer.encode_plus(
#         sn_sp,
#         add_special_tokens=False,
#         return_tensors='pt',
#         return_attention_mask=True,
#     )
#     word_ids = torch.Tensor(encoded.word_ids())

#     last_hidden = gpt2_model(encoded.input_ids).last_hidden_state

#     # aggregate the embeddings to word-level 
#     embeddings = aggregate_input_embeddings(
#         embeddings=last_hidden,
#         word_ids=word_ids,
#         aggregate=aggregate,
#     )

#     return embeddings, word_ids, sn_sp


def get_embeddings_seq2seq(
    data_instance: Dict[str, Any],
    tokenizer: transformers.GPT2TokenizerFast,
    gpt2_model: transformers.GPT2Model,
    bert_embeddings: torch.nn.Embedding,
    instance_idx: int,
    aggregate: str = 'mean', # 'mean', 'sum'
    max_length: int = 128,
    sp_pad_token: int = 127,
):
    """
    Get the embeddings of the scanpath (fixated words) from the encoder.
    :param data_instance: the data instance from the dataset.
    :param tokenizer: the tokenizer.
    :param gpt2_model: the GPT2 model.
    :param bert_embeddings: the BERT embeddings.
    :param aggregate: the aggregation method, either summing or averaging the sub-word embeddings.
    :return: the embeddings of the scanpath, the padded fixation durations, and the attention mask.
    """

    sn_repr_len = data_instance['sn_repr_len']

    # the sentence 
    sn_words = data_instance['words_for_mapping'].split()
    while sn_words[-1] == '[PAD]':
        sn_words.pop()
    # remove the CLS and SEP tokens 
    sn_words = sn_words[1:-1]
    # chinese characters are one string in a list
    if sp_pad_token == 67: # chinese pad token
        sn_words = list(sn_words[0])

    # the scanpath 
    # remove the CLS token 
    sp_ids = data_instance['sn_sp_repr'][sn_repr_len:][1:]
    # cut off the trailing pad tokens
    while sp_ids[-1] == sp_pad_token:
        sp_ids.pop()
    # remove the SEP token 
    sp_ids = sp_ids[:-1]

    # make the scanpath ids start from 0 for re-ordering of the embeddings
    sp_ids = [sp_id - 1 for sp_id in sp_ids]

    # get the scanpath as fixated words 
    sp_words = list()
    try:
        for sp_id in sp_ids:
            sp_words.append(sn_words[sp_id])
    except:
        print(f'Error at index {instance_idx}')
        breakpoint()
        return None, None, None
        

    # get the fixation durations 
    fix_durs = data_instance['sn_sp_fix_dur'][sn_repr_len+1:]
    while fix_durs[-1] == 0:
        fix_durs.pop()
    # convert to tensor 
    fix_durs = torch.Tensor(fix_durs)

    # get the sentence encoding 
    sn_enc = tokenizer.encode_plus(
        sn_words,
        add_special_tokens=False,
        return_tensors='pt',
        is_split_into_words=True,
    )
    sn_word_ids = torch.Tensor(sn_enc.word_ids())

    # get the embeddings 
    with torch.no_grad():
        last_hidden = gpt2_model(sn_enc.input_ids).last_hidden_state

    # aggregate the embeddings to word-level 
    sn_embeddings = aggregate_input_embeddings(
        embeddings=last_hidden,
        word_ids=sn_word_ids,
        aggregate=aggregate,
    )

    # convert sp_ids to tensor
    sp_ids = torch.Tensor(sp_ids).long()

    # re-order the embeddings as scanpath 
    sp_embeddings = sn_embeddings[:, sp_ids, :]

    # pad the embeddings and fixation durations to max input length
    # and get the attention mask 
    sp_embeddings_padded, fix_durs_padded, attention_mask = padding_and_mask_seq2seq(
        sp_embeddings=sp_embeddings,
        fix_durs=fix_durs,
        bert_embeddings=bert_embeddings,
        max_length=max_length,
    )

    return sp_embeddings_padded.squeeze(0), fix_durs_padded, attention_mask.squeeze(0)


def padding_and_mask_seq2seq(
    sp_embeddings: torch.Tensor,
    bert_embeddings: torch.nn.Embedding,
    max_length: int,
    fix_durs: Optional[torch.Tensor] = None,
    inference: Optional[bool] = None,
):
    """
    Add the BERT CLS token to the beginning of the scanpath embedding (needed for pooler output).
    Pad the scanpath embeddings and fixation durations to max input lenght.
    Use the PAD token embedding for padding.
    """
    # get the embedding for the pad token
    pad_emb = bert_embeddings(torch.Tensor([0]).long())
    cls_emb = bert_embeddings(torch.Tensor([101]).long())

    # prepend the cls emb to the sp_embeddings
    sp_embeddings = torch.cat((cls_emb.unsqueeze(0), sp_embeddings), dim=1)

    # pad the embeddings 
    current_length =sp_embeddings.size(1)
    padding_needed = max_length - current_length
    pad_tensor = pad_emb.unsqueeze(0).expand(1, padding_needed, -1)
    sp_embeddings_padded = torch.cat((sp_embeddings, pad_tensor), dim=1)

    # create attention mask
    sp_mask = torch.ones((1, current_length), dtype=torch.long)
    pad_mask = torch.zeros((1, padding_needed), dtype=torch.long)
    attention_mask = torch.cat((sp_mask, pad_mask), dim=1)

    if inference:
        return sp_embeddings_padded, attention_mask

    # prepend 0 to the fixation durations because the first word is the CLS token
    fix_durs = torch.cat((torch.Tensor([0]), fix_durs), dim=0)

    # pad the fixation durations
    fix_dur_pad = torch.zeros(padding_needed)
    fix_durs_padded = torch.cat((fix_durs, fix_dur_pad), dim=0)

    return sp_embeddings_padded, fix_durs_padded, attention_mask


def aggregate_input_embeddings(
    embeddings: torch.Tensor,
    word_ids: torch.Tensor,
    aggregate: str = 'mean',  # 'mean', 'sum'
):
    """
    Aggregate the embeddings that are input to the fixation module to word-level.
    :param embeddings: the last hidden state (contextualised embeddings) of the sentence-scanpath concatenation
        when passed through the GPT2 model.
    :param word_ids: the word ids of the sentence-scanpath concatenation.
    :param aggregate: the aggregation method, either summing or averaging the sub-word embeddings.
    :return: the aggregated word embeddings.
    """
    # get the unique indices and inverse
    unique_indices, inverse_indices = torch.unique(word_ids, return_inverse=True)

    # sum the tensor along the dimension 1 (sequence length) for the same word ids 
    summed_tensor = torch.zeros((1, unique_indices.size(0), embeddings.size(2)))
    summed_tensor = summed_tensor.scatter_add(1, inverse_indices.unsqueeze(0).unsqueeze(-1).expand_as(embeddings), embeddings)

    if aggregate == 'sum':
        return summed_tensor 
    
    elif aggregate == 'mean':

        # count the occurrences of each word id (how many sub-words per word)
        counts = torch.zeros(unique_indices.size(0)).scatter_add(0, inverse_indices, torch.ones_like(inverse_indices, dtype=torch.float))

        # average the summed tensor 
        averaged_tensor = summed_tensor / counts.view(1, -1, 1)
        return averaged_tensor


class Seq2SeqDataset(Dataset):
    def __init__(
        self, 
        data: Dict[str, torch.Tensor],
        normalize: Optional[bool] = None,
        inference: Optional[bool] = None,
        ):
        super().__init__()
        self.data = data
        self.normalize = normalize
        self.inference = inference

    def __len__(self):
        return len(self.data['sp_embeddings'])
    
    def __getitem__(self, idx):
        if self.inference:
            sample = {
                'sp_embeddings': self.data['sp_embeddings'][idx],
                'attention_masks': self.data['attention_masks'][idx],
            }
            return sample
        else:
            sample = {
                'sp_embeddings': self.data['sp_embeddings'][idx],
                'attention_masks': self.data['attention_masks'][idx],
                'fix_durs': self.data['fix_durs'][idx],
            }
            if self.normalize:
                sample['fix_durs_normalized'] = self.data['fix_durs_normalized'][idx]
            return sample
        

def prepare_seq2seq_data(
    data: datasets.DatasetDict,
    tokenizer: transformers.GPT2TokenizerFast,
    gpt2_model: transformers.GPT2Model,
    bert_embeddings: torch.nn.Embedding,
    aggregate: str = 'mean',
    max_length: int = 128,
    sp_pad_token: int = 127,
):
    """
    Prepare the data for training the fixation duration module.
    :param data: the dataset.
    :param tokenizer: the tokenizer.
    :param gpt2_model: the GPT2 model.
    :param bert_embeddings: the BERT embeddings.
    :param aggregate: the aggregation method, either summing or averaging the sub-word embeddings.
    :param max_length: the maximum input length.
    :return: the data for training the fixation duration module.
    """
    data_dict = {
        'sp_embeddings': [],
        'attention_masks': [],
        'fix_durs': [],
    }

    for idx, instance in tqdm(enumerate(data['train'])):

        sp_embeddings, fix_durs, attention_mask = get_embeddings_seq2seq(
            data_instance=instance,
            tokenizer=tokenizer,
            gpt2_model=gpt2_model,
            bert_embeddings=bert_embeddings,
            instance_idx=idx,
            aggregate=aggregate,
            max_length=max_length,
            sp_pad_token=sp_pad_token,
        )
        if sp_embeddings is None:
            continue

        data_dict['sp_embeddings'].append(sp_embeddings)
        data_dict['attention_masks'].append(attention_mask)
        data_dict['fix_durs'].append(fix_durs)

    return data_dict


def split_train_val_data(
    data: Dict[str, List[torch.Tensor]],
    val_size: float = 0.1,
):
    """
    Split the train data into train and validation data.
    :param data: the data.
    :param val_size: the size of the validation data.
    :return: the train and validation data.
    """
    num_samples = len(next(iter(data.values())))
    # shuffle the indices
    indices = list(range(num_samples))
    random.shuffle(indices)

    # compute the split point 
    split_point = int(num_samples * val_size)
    train_indices = indices[split_point:]
    val_indices = indices[:split_point]

    train_data = {key: [value[i] for i in train_indices] for key, value in data.items()}
    val_data = {key: [value[i] for i in val_indices] for key, value in data.items()}

    return train_data, val_data


def get_embeddings_seq2seq_hp(
    sn_repr_len: int,
    sn_words: List[str],
    sp_ids: List[int],
    tokenizer: transformers.GPT2TokenizerFast,
    gpt2_model: transformers.GPT2Model,
    bert_embeddings: torch.nn.Embedding,
    aggregate: str = 'mean',
    max_length: int = 128,
    sp_pad_token: int = 127,
):
    """
    Get the embeddings of the scanpath (fixated words) from the encoder.
    :param sn_repr_len: the length of the sentence representation.
    :param sn_words: the words of the sentence.
    :param sp_ids: the scanpath ids.
    :param tokenizer: the tokenizer.
    :param gpt2_model: the GPT2 model.
    :param bert_embeddings: the BERT embeddings.
    :param aggregate: the aggregation method, either summing or averaging the sub-word embeddings.
    :return: the embeddings of the scanpath, the padded fixation durations, and the attention mask.
    """

    pad_idx = [i for i, word in enumerate(sn_words) if word == '[PAD]']
    sep_idx = [sn_words.index('[SEP]')]
    all_remove_idx = [0]  # for CLS 
    all_remove_idx += sep_idx 
    all_remove_idx += pad_idx

    # get rid of trailing pad tokens in sentence 
    while sn_words[-1] == '[PAD]':
        sn_words.pop()
    # get rid of the CLS and SEP tokens
    sn_words = sn_words[1:-1]

    # the scanpath 
    # get rid of predicted CLS, SEP and wrongly predicted PAD tokens (will throw error)
    sp_ids = [sp_id for sp_id in sp_ids if sp_id not in all_remove_idx]

    # make the scanpath ids start from 0 for re-ordering of the embeddings
    sp_ids = [sp_id - 1 for sp_id in sp_ids]

    # get the scanpath as fixated words
    sp_words = list()
    for sp_id in sp_ids:
        sp_words.append(sn_words[sp_id])
    
    # get the sentence encoding 
    sn_enc = tokenizer.encode_plus(
        sn_words,
        add_special_tokens=False,
        return_tensors='pt',
        is_split_into_words=True,
    )
    sn_word_ids = torch.Tensor(sn_enc.word_ids())

    # get the embeddings 
    with torch.no_grad():
        last_hidden = gpt2_model(sn_enc.input_ids).last_hidden_state

    # aggregate the embeddings to word-level
    sn_embeddings = aggregate_input_embeddings(
        embeddings=last_hidden,
        word_ids=sn_word_ids,
        aggregate=aggregate,
    )

    # convert sp_ids to tensor
    sp_ids = torch.Tensor(sp_ids).long()

    # re-order the embeddings as scanpath
    sp_embeddings = sn_embeddings[:, sp_ids, :]

    # pad the embeddings to max input length and get the attention mask
    sp_embeddings_padded, attention_mask = padding_and_mask_seq2seq(
        sp_embeddings=sp_embeddings,
        bert_embeddings=bert_embeddings,
        max_length=max_length,
        inference=True,
    )

    return sp_embeddings_padded.squeeze(0), attention_mask.squeeze(0)





def prepare_seq2seq_data_hp(
    scandl_output: Dict[str, Any],
    tokenizer: transformers.GPT2TokenizerFast,
    gpt2_model: transformers.GPT2Model,
    bert_embeddings: torch.nn.Embedding,
    aggregate: str = 'mean',
    max_length: int = 128,
    sp_pad_token: int = 127,
):
    """
    Prepare the scandl output for inference of the hyper-parameter search of the Seq2Seq fixation duration model.
    :param scandl_output: the ScanDL output.
    :return: the data for inference.
    """
    data_dict = {
        'sp_embeddings': [],
        'attention_masks': [],
        'original_fix_durs': [],
        'predicted_sp_ids': [],
        'reader_ids': [],
        'sn_ids': [],
    }

    for idx in tqdm(range(len(scandl_output['predicted_sp_ids']))):

        sn_repr_len = scandl_output['sn_repr_len'][idx]
        if sp_pad_token == 67:
            # for Chinese: make sure the words are split correctly (chinese characters have no whitespace)
            sn_words = scandl_output['words_for_mapping'][idx].split()
            sn_words = [sn_words[0]] + list(sn_words[1]) + sn_words[2:] 
        else:
            sn_words = scandl_output['words_for_mapping'][idx].split()
        sp_ids = scandl_output['predicted_sp_ids'][idx]


        try:
            sp_embeddings, attention_mask = get_embeddings_seq2seq_hp(
                sn_repr_len=sn_repr_len,
                sn_words=sn_words,
                sp_ids=sp_ids,
                tokenizer=tokenizer,
                gpt2_model=gpt2_model,
                bert_embeddings=bert_embeddings,
                aggregate=aggregate,
                max_length=max_length,
                sp_pad_token=sp_pad_token,
            )

            # get the original fixation durations
            fix_durs = scandl_output['sn_sp_fix_dur'][idx][sn_repr_len:]
            while fix_durs[-1] == 0:
                fix_durs.pop()
            fix_durs.append(0)

            data_dict['sp_embeddings'].append(sp_embeddings)
            data_dict['attention_masks'].append(attention_mask)
            data_dict['original_fix_durs'].append(str(fix_durs))
            data_dict['predicted_sp_ids'].append(str(sp_ids))
            data_dict['reader_ids'].append(scandl_output['reader_ids'][idx])
            data_dict['sn_ids'].append(scandl_output['sn_ids'][idx])
        except:
            print(f'Error at index {idx}')
            continue

    
    return data_dict


class Seq2SeqDatasetHP(Dataset):
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
            'attention_masks': self.data['attention_masks'][idx],
            #'predicted_sp_words': self.data['predicted_sp_words'][idx],
            #'original_sp_words': self.data['original_sp_words'][idx],
            'predicted_sp_ids': self.data['predicted_sp_ids'][idx],
            # 'original_sp_ids': self.data['original_sp_ids'][idx],
            # 'original_sn': self.data['original_sn'][idx],
            'sn_ids': self.data['sn_ids'][idx],
            'reader_ids': self.data['reader_ids'][idx],
            # 'sn_repr_len': self.data['sn_repr_len'][idx],
            # 'words_for_mapping': self.data['words_for_mapping'][idx],
            # 'sn_sp_repr': self.data['sn_sp_repr'][idx],
            # 'sn_sp_fix_dur': self.data['sn_sp_fix_dur'][idx],
            'original_fix_durs': self.data['original_fix_durs'][idx],
        }
        return sample