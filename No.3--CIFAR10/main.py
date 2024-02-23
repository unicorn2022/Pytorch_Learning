import torch
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
    optim = torch.optim.SGD(model.parameters(), lr=0.01)

    # writer = SummaryWriter('./logs')
    # step = 0
    for epoch in range(20):
        sum_loss = 0
        for data in dataloader:
            img, target = data
            # 模型推理
            output = model(img)
            # 计算损失
            result_loss = loss(output, target)
            sum_loss += result_loss.item()
            # 反向传播
            optim.zero_grad()
            result_loss.backward()
            optim.step()

            # writer.add_images('input', img, step)
            # writer.add_images('output', output, step)
            # step += 1
        print('epoch: {}, sum_loss: {}'.format(epoch, sum_loss))
    # writer.close()