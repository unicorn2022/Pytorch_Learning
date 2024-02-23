import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(196608, 10)

    def forward(self, x):
        x = self.linear(x)
        return x