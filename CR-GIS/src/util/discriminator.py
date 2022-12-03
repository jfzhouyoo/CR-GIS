import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Discriminator(nn.Module):
    def __init__(self, hidden_size):
        super(Discriminator, self).__init__()
        self.bilinear = nn.Bilinear(hidden_size, hidden_size, 1)
        self.sigm = nn.Sigmoid()
        for m in self.modules():
            self.weights_init(m)
    
    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    
    def forward(self, prior_enc, poster_enc):
        prior_enc = self.sigm(prior_enc)
        poster_enc = self.sigm(poster_enc)
        score = torch.squeeze(self.bilinear(prior_enc, poster_enc), 1)
        return self.sigm(score)
    
    def get_reward(self, prior_enc, poster_enc):
        score = self.forward(prior_enc, poster_enc)
        discriminator_reward = torch.log(score+1e-8) - torch.log(1-score+1e-8)
        return discriminator_reward.detach()