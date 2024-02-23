import torch
import torch.nn as nn


input = torch.tensor([0.1, 0.2, 0.3]).view(1, 3)
target = torch.tensor([1])
loss = nn.CrossEntropyLoss()(input, target)
print(loss)