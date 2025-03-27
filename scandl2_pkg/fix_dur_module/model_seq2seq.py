


import torch 
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertEncoder
from typing import Optional


class Seq2SeqModel(nn.Module):

    def __init__(
        self,
        config,
        output_dim,
        num_linear,
        dropout,
    ):
        super().__init__()
        self.config = config
        self.output_dim = output_dim
        self.encoder = BertEncoder(config)
        self.pooler = Pooler(config)

        layers_list = list()
        for i in range(num_linear):
            layers_list.append(nn.Linear(config.hidden_size, config.hidden_size))
            layers_list.append(nn.ReLU())
            layers_list.append(nn.Dropout(dropout))
        self.ff = nn.Sequential(*layers_list)
        #self.ff = nn.Linear(config.hidden_size, config.hidden_size)
        self.ff_out = nn.Linear(config.hidden_size, output_dim)

    
    def _invert_attention_mask(self, attention_mask):
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float32).min
        return extended_attention_mask

    
    def forward(
        self,
        sp_embeddings,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
    ):
        # get the extended attention mask 
        # zeros and ones are inverted such that what is not maked is 0 and what is masked is -inf 
        if attention_mask is not None:
            attention_mask = self._invert_attention_mask(attention_mask)
            encoder_outputs = self.encoder(
                sp_embeddings, 
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )
        else:
            encoder_outputs = self.encoder(
                sp_embeddings, 
                output_attentions=output_attentions,
            )

        last_hidden_state = encoder_outputs.last_hidden_state
        
        
        # pool the encoder output: the hidden state of the CLS token is passed through another linear layer
        pooled_output = self.pooler(last_hidden_state)
        
        # map to the output dimension
        out = self.ff(pooled_output)
        out = self.ff_out(out)

        if output_attentions:
            attentions = encoder_outputs.attentions
            return out, attentions

        else:
            return out


class Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
    
    def forward(self, hidden_states):
        # pool the output by taking the hidden state of the first token (the CLS token)
        # and pass it through another linear layer wtih tanh activation 
        cls_out = hidden_states[:, 0]
        pooled_output = self.dense(cls_out)
        pooled_output = self.activation(pooled_output)
        return pooled_output