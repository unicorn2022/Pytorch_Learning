import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # input: 3 @ 32 x 32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        # feature map: 32 @ 32 x 32
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # feature map: 32 @ 16 x 16
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        # feature map: 32 @ 16 x 16
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # feature map: 32 @ 8 x 8
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        # feature map: 64 @ 8 x 8
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        # feature map: 64 @ 4 x 4
        self.flatten = nn.Flatten()
        # hidden units: 1024
        self.linear1 = nn.Linear(in_features=1024, out_features=64)
        # hidden units: 64
        self.linear2 = nn.Linear(in_features=64, out_features=10)
        # output: 10

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x
    
if __name__ == '__main__':
    model = MyModel()
    print(model)
    
    # 检验模型是否合法, 输入为: (batch_size, 3, 32, 32)
    input = torch.ones((64, 3, 32, 32))
    output = model(input)
    # 输出应为: (batch_size, 10)
    print(output.shape)

    # 输出模型
    writer = SummaryWriter('./logs')
    writer.add_graph(model=model, input_to_model=input)
    writer.close()