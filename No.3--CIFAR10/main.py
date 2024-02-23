import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from nn_linear import MyModel


if __name__ == '__main__':
    dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=torchvision.transforms.ToTensor(), download=True)
    dataloader = DataLoader(dataset, batch_size=64)
    model = MyModel()

    # writer = SummaryWriter('./logs')
    # step = 0
    for data in dataloader:
        img, target = data
        print(img.shape)
        img = torch.flatten(img)
        print(img.shape)
        output = model(img)
        print(output.shape)
        break

        # writer.add_images('input', img, step)
        # writer.add_images('output', output, step)
        # step += 1
    # writer.close()