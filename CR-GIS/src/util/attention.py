#!/usr/bin/env python

import torch
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self,
                 query_size,
                 memory_size=None,
                 hidden_size=None,
                 mode="mlp",
                 return_attn_only=False,
                 project=False,
                 device=None):
        super(Attention, self).__init__()
        assert (mode in ["dot", "general", "mlp"]), (
            "Unsupported attention mode: {mode}"
        )

        self.query_size = query_size
        self.memory_size = memory_size or query_size
        self.hidden_size = hidden_size or query_size
        self.mode = mode
        self.return_attn_only = return_attn_only
        self.project = project
        self.device = device

        if mode == "general":
            self.linear_query = nn.Linear(
                self.query_size, self.memory_size, bias=False)
        elif mode == "mlp":
            self.linear_query = nn.Linear(
                self.query_size, self.hidden_size, bias=True)
            self.linear_memory = nn.Linear(
                self.memory_size, self.hidden_size, bias=False)
            self.tanh = nn.Tanh()
            self.v = nn.Linear(self.hidden_size, 1, bias=False)

        self.softmax = nn.Softmax(dim=-1)

        if self.project:
            self.linear_project = nn.Sequential(
                nn.Linear(in_features=self.hidden_size + self.memory_size,
                          out_features=self.hidden_size),
                nn.Tanh())

    def __repr__(self):
        main_string = "Attention({}, {}".format(self.query_size, self.memory_size)
        if self.mode == "mlp":
            main_string += ", {}".format(self.hidden_size)
        main_string += ", mode='{}'".format(self.mode)
        if self.project:
            main_string += ", project=True"
        main_string += ")"
        return main_string

    def forward(self, query, memory, mask=None):
        if self.mode == "dot":
            assert query.size(-1) == memory.size(-1)
            attn = torch.bmm(query, memory.transpose(1, 2))
        elif self.mode == "general":
            assert self.memory_size == memory.size(-1)
            key = self.linear_query(query)
            attn = torch.bmm(key, memory.transpose(1, 2))
        else:
            hidden = self.linear_query(query).unsqueeze(
                2) + self.linear_memory(memory).unsqueeze(1)
            key = self.tanh(hidden)
            attn = self.v(key).squeeze(-1)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, query.size(1), 1)
            attn.masked_fill_(mask, -float("inf"))

        weights_tmp = self.softmax(attn)   # need avoid the problem of "all -inf"
        weights = weights_tmp.clone()  # fix a bug, the attn_score may be all -inf
        weights[torch.isnan(weights) == True] = torch.tensor(0, dtype=torch.float, device=self.device)

        weighted_memory = torch.bmm(weights, memory)

        if self.return_attn_only:
            return weights

        if self.project:
            project_output = self.linear_project(
                torch.cat([weighted_memory, query], dim=-1))
            return project_output, weights
        else:
            return weighted_memory, weights


class SelfAttentionLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_size, alpha=0.2, dropout=0.5):
        super(SelfAttentionLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.dropout = dropout
        
        self.Wa = nn.Parameter(torch.zeros(size=(self.embedding_dim, self.hidden_size)))
        self.Wb = nn.Parameter(torch.zeros(size=(self.hidden_size, 1)))
        nn.init.xavier_uniform_(self.Wa.data, gain=1.414)
        nn.init.xavier_uniform_(self.Wb.data, gain=1.414)

    def forward(self, hidden, mask):
        '''
        shape of hidden: [batch_size, entity_num, embedding_dim]
        shape of mask: [batch_size, entity_num], 0/1
        '''
        
        N = hidden.shape[0] # batch_size
        assert self.embedding_dim == hidden.shape[2]

        attention_weight = torch.matmul(torch.tanh(torch.matmul(hidden, self.Wa)), self.Wb).squeeze() # shape: [batch_size, entity_num]
        attention_weight = attention_weight * mask.float() # shape: [batch_size, entity_num]
        attention_weight /= torch.sum(attention_weight, -1).unsqueeze(-1) + 1e-30 # shape: [batch_size, entity_num]
        attention_hidden = attention_weight.unsqueeze(-1) * hidden # shape: [batch_size, entity_num, embedding_dim]
        output = torch.sum(attention_hidden, 1) # shape: [batch_size, embedding_dim]

        return output