import torch
import torch.nn as nn
import torch.nn.functional as F

input = torch.tensor(
    [[1, 2, 0, 3, 1],
    [0, 1, 2, 3, 1],
    [1, 2, 1, 0, 0],
    [5, 2, 3, 1, 1],
    [2, 1, 0, 1, 1]])

kernel = torch.tensor(
    [[1, 2, 1],
    [0, 1, 0],
    [2, 1, 0]])

input = input.view(1, 1, 5, 5)

kernel = kernel.view(1, 1, 3, 3)

output = F.conv2d(
    input=input,   # 输入:     (batch_size, in_channel, input_H, input_W)
    weight=kernel, # 卷积核:   (out_channel, in_channel/groups, kernel_H, kernel_W)
    stride=1,      # 步长
    padding=1,     # 填充
)

print(output)