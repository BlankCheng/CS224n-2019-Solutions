import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

reference = [['love', 'can', 'always', 'find', 'a', 'way']]
candidate = ['the', 'love', 'can', 'always', 'do']
print(np.exp(0.5*np.log(0.6)+0.5*np.log(0.5)) * np.exp(-0.2))
score = sentence_bleu(reference, candidate, weights=(0.5,0.5,0,0))
print(score)

# test attn
# W = nn.Linear(50, 100)
# dec_hidden = torch.rand(4, 50)
# enc_hiddens_proj = torch.rand(4, 50, 15)
# e_t = torch.bmm(dec_hidden.unsqueeze(1), enc_hiddens_proj)
# print(e_t.size())



# test decode
# def step(Y, dec):
#     return (torch.rand(4, 50), torch.rand(4, 50)), torch.rand(4, 50), torch.rand(4, 50)
#
# combined_outputs = []
# enc_hiddens_proj = torch.rand(4, 10, 50)  # (b, src_len, h)
# dec_state = (torch.rand(4, 50), torch.rand(4, 50))
# o_prev = torch.zeros(4, 50)
#
# Y = torch.rand(15, 4, 50)
# for Y_t in torch.split(Y, 1):
#     Y_t = torch.squeeze(Y_t, 0)
#     Ybat_t = torch.cat((Y_t, o_prev), -1)
#     dec_state, o_t, e_t = step(Ybat_t, dec_state)
#     combined_outputs.append(o_t)
#     o_prev = o_t
# combined_outputs = torch.stack(combined_outputs, 0)
# print(combined_outputs.size())

# test for h projection
# h_last = torch.tensor([[[1, 1, 1, 1],
#                         [3, 3, 3, 3],
#                         [5, 5, 5, 5]],
#                        [[11, 11, 11, 11],
#                         [13, 13, 13, 13],
#                         [2, 2, 2, 2]]])
# print(h_last.permute(1, 0, 2).reshape(3, 8))
# print(h_last.reshape(3, 8))


# test pad and pack
# X = torch.rand(10, 4, 50)  # (src_len, b, e)
# encoder = nn.LSTM(input_size=50,
#                        hidden_size=50,
#                        num_layers=1,
#                        bias=True,
#                        batch_first=False,
#                        dropout=0,
#                        bidirectional=True)
# source_lengths = [10, 7, 3, 2]
# packed = pack_padded_sequence(X, source_lengths)
# print(packed)
# print(packed.data)
# print(packed.data.size())
# enc_hiddens = encoder(packed)
# print(pad_packed_sequence(enc_hiddens[0])[0])
# print(enc_hiddens[1][0].size())
# print(enc_hiddens[1][1].size())
# print(encoder(X)[0])
# print(encoder(X)[1][0].size())
# print(encoder(X)[1][1].size())



