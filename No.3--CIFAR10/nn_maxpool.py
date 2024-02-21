import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=True)
    
    def forward(self, x):
        x = self.maxpool1(x)
        return x
    
if __name__ == '__main__':
    dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=torchvision.transforms.ToTensor(), download=True)
    dataloader = DataLoader(dataset, batch_size=64)
    model = MyModel()

    writer = SummaryWriter('./logs')
    step = 0
    for data in dataloader:
        img, target = data
        output = model(img)

        writer.add_images("input", img, step)
        writer.add_images("output", output, step)
        step += 1
    writer.close()