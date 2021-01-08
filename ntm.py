import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from loss import EntropyLoss
import math

class NTMModel(nn.Module):
    def __init__(self, hidden_dim=128, vocab_size=20, topic_size=10):
        super(NTMModel, self).__init__()
        self.vocab_size = vocab_size
        self.topic_size = topic_size
        self.hidden_dim = hidden_dim
        
        self.hidden_encoder = torch.nn.Linear(vocab_size, hidden_dim)
        self.relu1 = torch.nn.ReLU()
        self.pi = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = torch.nn.ReLU()
        self.mu_encoder = torch.nn.Linear(hidden_dim, topic_size)
        self.logvar_encoder = torch.nn.Linear(hidden_dim, topic_size)
        self.epsilon = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        
        self.R = torch.nn.Linear(topic_size, vocab_size, bias = True)
        self.softmax = torch.nn.Softmax(dim = 1)
        
    def _reset_params(self, initializer):
        for child in self.children():
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
        
    def get_topic_word_distribution(self):
        return self.R.weight

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels = None):
        sequence_output = self.hidden_encoder(input_ids)
        sequence_output = self.relu1(sequence_output)
        sequence_output = self.pi(sequence_output)
        sequence_output = self.relu2(sequence_output)
        sequence_output_mu = self.mu_encoder(sequence_output).to(sequence_output.device)
        sequence_output_logvar = self.logvar_encoder(sequence_output).to(sequence_output.device)
        epsilon_sample = self.epsilon.sample(sequence_output_logvar.shape).squeeze()
        epsilon_sample = torch.tensor(epsilon_sample, dtype = torch.half).to(sequence_output.device)
        
        std_dev = torch.sqrt(torch.exp(sequence_output_logvar)).to(sequence_output.device)
        
        sequence_output_h = (sequence_output_mu + torch.mul(std_dev, epsilon_sample)).to(sequence_output.device)
        
        sequence_output = self.R(sequence_output_h)
        sequence_output = self.softmax(sequence_output)
        sequence_output = torch.squeeze(sequence_output)
        
        kl_divergence = -0.5 * torch.sum(1.0 + sequence_output_logvar - torch.square(sequence_output_mu) - torch.exp(sequence_output_logvar), dim = 1)
        likehood = -torch.sum(torch.mul(torch.log(sequence_output + 1e-6), input_ids), dim = 1)
        loss = torch.mean(kl_divergence + likehood)
        
        return loss
