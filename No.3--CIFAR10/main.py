import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from nn_CIFAR10 import MyModel


if __name__ == '__main__':
    dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=transforms.ToTensor(), download=True)
    dataloader = DataLoader(dataset, batch_size=64)
    model = MyModel()
    loss = nn.CrossEntropyLoss()

    # writer = SummaryWriter('./logs')
    # step = 0
    for data in dataloader:
        img, target = data
        output = model(img)
        result_loss = loss(output, target)
        result_loss.backward()
        print(result_loss)

        # writer.add_images('input', img, step)
        # writer.add_images('output', output, step)
        # step += 1
    # writer.close()