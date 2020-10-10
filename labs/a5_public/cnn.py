#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils

### YOUR CODE HERE for part 1i
class CNN(nn.Module):
    def __init__(self, embed_size, output_size, k=5):
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(embed_size, output_size, k)

    def forward(self, x_reshaped):
        """
        x_reshaped: (b*s_len, e, w_len)
        @return: (b*s_len, e)
        """
        x_conv = self.conv(x_reshaped)
        x_conv_out = torch.max(F.relu(x_conv), dim=-1)[0]
        return x_conv_out
### END YOUR CODE

if __name__ == "__main__":
    embed_size = 50
    output_size = 50
    x = torch.randn(2, 50, 21)
    net = CNN(embed_size, output_size)
    out = net(x)
    print(out.size())