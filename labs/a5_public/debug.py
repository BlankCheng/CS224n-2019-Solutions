from docopt import docopt
import torch.nn as nn
import torch

batch_size = 10
start_idx = 2
current_char = torch.tensor([[start_idx] * batch_size])  # idx of '<start>' token
initialChars = torch.ones(1, batch_size, dtype=torch.long) * start_idx
print(current_char)
print(initialChars)