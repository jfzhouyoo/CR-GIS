import math
import torch
import datetime
import numpy as np
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from util.attention import Attention

class SGNN(nn.Module):
    def __init__(self, embedding_dim, layers, device):
        super(SGNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.layers = layers
        self.device = device

        self.input_size = 2 * self.embedding_dim
        self.gate_size = 3 * self.embedding_dim

        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.embedding_dim))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.embedding_dim))
        self.b_oah = Parameter(torch.Tensor(self.embedding_dim))

        self.linear_edge_in = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.linear_edge_out = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.linear_edge_f = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)

        self.attention = Attention(query_size=self.embedding_dim,
                                   memory_size=self.embedding_dim,
                                   hidden_size=self.embedding_dim,
                                   mode="dot",
                                   device=self.device)
        self.tanh = nn.Tanh()
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def average_pooling(self, entities_feature, graph_mask):
        entity_num = torch.sum(graph_mask, 1)
        entities_feature = entities_feature * graph_mask.unsqueeze(-1).float()
        output = torch.sum(entities_feature, 1) / (entity_num.unsqueeze(-1).float() + 1e-30)
        return output

    def GNNCell(self, entities_feature, adjacent_matrix):
        adjacent_matrix = adjacent_matrix.float()
        input_in = torch.matmul(adjacent_matrix[:, :, :adjacent_matrix.shape[1]], self.linear_edge_in(entities_feature)) + self.b_iah
        input_out = torch.matmul(adjacent_matrix[:, :, adjacent_matrix.shape[1]: 2 * adjacent_matrix.shape[1]], self.linear_edge_out(entities_feature)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        g_input = F.linear(inputs, self.w_ih, self.b_ih)
        g_hidden = F.linear(entities_feature, self.w_hh, self.b_hh)
        input_r, input_i, input_n = g_input.chunk(3, 2)
        hidden_r, hidden_i, hidden_n = g_hidden.chunk(3, 2)
        reset_gate = torch.sigmoid(input_r + hidden_r)
        input_gate = torch.sigmoid(input_i + hidden_i)
        new_gate = torch.tanh(input_n + reset_gate * hidden_n)

        updated_entities_feature = new_gate + input_gate * (entities_feature - new_gate)
        return updated_entities_feature

    def forward(self, entities_feature, adjacent_matrix, graph_mask):
        star_node_hidden = self.average_pooling(entities_feature, graph_mask) # obtain star node
        for l in range(self.layers):
            entities_feature = self.GNNCell(entities_feature, adjacent_matrix)
            similarity = torch.matmul(entities_feature, star_node_hidden.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.embedding_dim) # shape of star node:[batch_size, embedding_dim], shape of similarity:[batch_size, entity_num]
            alpha = torch.sigmoid(similarity).unsqueeze(-1) # shape: [batch_size,entity_num,1]
            batch_size, entity_num = entities_feature.shape[0], entities_feature.shape[1]
            star_node_hidden_repeat = star_node_hidden.repeat(1, entity_num).view(batch_size, entity_num, self.embedding_dim)
            entities_feature = (1-alpha) * entities_feature + alpha * star_node_hidden_repeat
            # attention: update star node
            weigted_entities, entities_attn = self.attention(query=star_node_hidden.unsqueeze(1), 
                                                             memory=entities_feature, mask=graph_mask.eq(0))
            star_node_hidden = weigted_entities.squeeze(1)
        return entities_feature, star_node_hidden
