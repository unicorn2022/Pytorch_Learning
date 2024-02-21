import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output
    
if __name__ == '__main__':
    net = Network()
    input = torch.tensor(1.0)
    output = net(input)
    print(output)