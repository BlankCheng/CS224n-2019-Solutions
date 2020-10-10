#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F


### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    def __init__(self, embed_size):
        super(Highway, self).__init__()
        self.W_proj = nn.Linear(embed_size, embed_size)
        self.W_gate = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x_conv_out):
        """
        x_conv_out: (b, s_len, e)
        @return: (b, s_len, e)
        """
        x_proj = F.relu(self.W_proj(x_conv_out))
        x_gate = F.sigmoid(self.W_gate(x_conv_out))
        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out
        x_word_emb = self.dropout(x_highway)
        return x_word_emb
### END YOUR CODE

if __name__ == "__main__":
    embed_size = 50
    x = torch.randn(2, 10, 50)
    net = Highway(embed_size)
    out = net(x)
    print(out.size())
