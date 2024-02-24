import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

dataset_path = '../No.3--CIFAR10/dataset'
dataset_train = datasets.CIFAR10(root=dataset_path, train=True, download=True)


# 载入预训练模型
vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
# 在已有模型后面添加层
# vgg16.add_module('on_CIFAR10', nn.Linear(1000, 10))
# 修改已有模型中的层
# vgg16.classifier[6] = nn.Linear(4096, 10)


# 保存&加载模型(pth): 保存结构+参数
torch.save(vgg16, 'vgg16.pth')
model = torch.load('vgg16.pth')

# 保存&加载模型(state_dict): 保存参数
torch.save(vgg16.state_dict(), 'vgg16_state_dict.pth')
model = models.vgg16().load_state_dict(torch.load('vgg16_state_dict.pth'))

writer = SummaryWriter('logs')
writer.add_graph(vgg16, torch.rand(1, 3, 224, 224))
writer.close()