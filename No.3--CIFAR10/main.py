import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from nn_CIFAR10 import MyModel


if __name__ == '__main__':
    dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=transforms.ToTensor(), download=True)
    dataloader = DataLoader(dataset, batch_size=64)
    model = MyModel()

    writer = SummaryWriter('./logs')
    step = 0
    for data in dataloader:
        img, target = data
        print(img.shape)
        img = torch.flatten(img)
        print(img.shape)
        output = model(img)
        print(output.shape)

        writer.add_images('input', img, step)
        writer.add_images('output', output, step)
        step += 1
    writer.close()