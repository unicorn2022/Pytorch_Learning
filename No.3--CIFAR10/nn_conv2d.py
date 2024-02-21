import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0 )
    
    def forward(self, x):
        x = self.conv1(x)
        return x

if __name__ == '__main__':
    dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=torchvision.transforms.ToTensor(), download=True)
    dataloader = DataLoader(dataset, batch_size=64)
    model = MyModel()

    writer = SummaryWriter('./logs')
    step = 0
    for data in dataloader:
        # img: [64, 3, 32, 32]
        img, target = data
        # output: [64, 6, 30, 30]
        output = model(img)
        # output: [xxx, 3, 30, 30]
        output = torch.reshape(output, (-1, 3, 30, 30))

        writer.add_images('input', img, step)
        writer.add_images('output', output, step)
        step += 1