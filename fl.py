import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from loss import EntropyLoss
import math
from pytorch_transformers import BertPreTrainedModel,BertModel

class BaselineModel(BertPreTrainedModel):
    def __init__(self, config, num_labels=3):
        super(BaselineModel, self).__init__(config)
        self.config = config
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels)

    def _reset_params(self, initializer):
        for child in self.children():
            if type(child) == BertModel:
                continue
            for p in child.parameters():
                if p.requires_grad:
                    if len(p.shape) > 1:
                        initializer(p)
                    else:
                        stdv = 1. / math.sqrt(p.shape[0])
                        torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def get_valid_seq_output(self, sequence_output, valid_ids):
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=sequence_output.dtype, device=sequence_output.device)
        for i in range(batch_size):
            temp = sequence_output[i][valid_ids[i] == 1]
            valid_output[i][:temp.size(0)] = temp
        return valid_output

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels = None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        tag_seq = torch.argmax(logits, -1)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return tag_seq,loss
        else:
            return tag_seq

class TMN(nn.Module):
    def __init__(self, topic_model_path, hidden_size, topic_method=0):
        super(TMN, self).__init__()
        self.topic_method = topic_method
        self.hidden_size = hidden_size
        w = torch.load(topic_model_path, map_location="cpu").float()
        self.vocab_size, self.topic_size = w.size()
        w = torch.cat((torch.zeros([2,self.topic_size]),w), 0)
        self.vocab_size += 2
        self.key_emb = nn.Embedding.from_pretrained(w)
        # self.val_emb = nn.Embedding.from_pretrained(w.t())
        self.val_emb = nn.Embedding(self.topic_size, hidden_size)
        self.src_topic_dense = nn.Linear(1, hidden_size)
        self.tgt_topic_dense = nn.Linear(hidden_size, hidden_size)
        self.hidden_state_dense = nn.Linear(hidden_size, hidden_size)
        self.softmax = torch.nn.Softmax(dim = 1)
        self.activation = nn.Tanh()
        # self.tgt_topic = self.ret_tgt_topic()

    def ret_tgt_topic(self):
        topic_input_ids = torch.tensor(list(range(self.topic_size)))
        return self.activation(self.tgt_topic_dense(self.val_emb(topic_input_ids)))

    def ret_topics_prob_w(self,  input_ids):
        output = self.key_emb(input_ids).sum(dim=-2)
        return self.softmax(output)

    def ret_topics_prob_d(self, input_ids, hidden_state):
        def get_input_len():
            batch_size,seq_len = input_ids.size()
            input_len_list = []
            for i in range(batch_size):
                input_len = seq_len
                input_padded_mask_indices = (input_ids[i] == 0).nonzero()
                if len(input_padded_mask_indices) != 0:
                    input_len = input_padded_mask_indices[0].item()
                input_len_list.append([input_len])
            return torch.tensor(input_len_list).to(input_ids.device)
        topic_output = (self.key_emb(input_ids).sum(dim=-2) / get_input_len()).unsqueeze(-1)
        topic_output = self.activation(self.src_topic_dense(topic_output))
        hidden_state = self.activation(self.hidden_state_dense(hidden_state))
        return self.softmax(torch.mul(topic_output, hidden_state.unsqueeze(1)).sum(2))

    def forward(self, input_ids, hidden_state):
        if self.topic_method == 0:
            topic_prob = self.ret_topics_prob_d(input_ids, hidden_state)
        elif self.topic_method == 1:
            topic_prob = self.ret_topics_prob_w(input_ids)
        else:
            topic_prob = self.ret_topics_prob_d(input_ids, hidden_state) + self.ret_topics_prob_w(input_ids)

        batch_size,seq_len = input_ids.size()
        topic_input_ids = torch.tensor(list(range(self.topic_size))).to(input_ids.device)
        topic_input_ids = torch.stack([topic_input_ids]*batch_size, 0)
        tgt_topic = self.tgt_topic_dense(self.val_emb(topic_input_ids))
        o = torch.mul(tgt_topic, topic_prob.unsqueeze(-1)).sum(dim=-2)
        return torch.add(o, hidden_state)


class TMNModel(BertPreTrainedModel):
    def __init__(self, config, num_labels=3, topic_model_path=None, topic_method=0):
        super(TMNModel, self).__init__(config)
        self.config = config
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels)
        self.domain_mem = TMN(topic_model_path, config.hidden_size, topic_method=topic_method)

    def _reset_params(self, initializer):
        for child in self.children():
            if type(child) == BertModel:
                continue
            for p in child.parameters():
                if p.requires_grad:
                    if len(p.shape) > 1:
                        initializer(p)
                    else:
                        stdv = 1. / math.sqrt(p.shape[0])
                        torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def get_valid_seq_output(self, sequence_output, valid_ids):
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=sequence_output.dtype, device=sequence_output.device)
        for i in range(batch_size):
            temp = sequence_output[i][valid_ids[i] == 1]
            valid_output[i][:temp.size(0)] = temp
        return valid_output

    def forward(self, input_ids, w_input_ids, token_type_ids=None, attention_mask=None, labels = None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.domain_mem(w_input_ids, pooled_output)
        logits = self.classifier(pooled_output)
        tag_seq = torch.argmax(logits, -1)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return tag_seq,loss
        else:
            return tag_seq
