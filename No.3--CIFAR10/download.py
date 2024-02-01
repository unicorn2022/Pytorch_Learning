import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

dataset_transform = transforms.Compose([
    transforms.ToTensor()
])

# 使用torchvision数据集, 如果没有则下载
train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=dataset_transform, download=True)

# 将数据集显示到tensorboard中
writter = SummaryWriter('./logs')
for i in range(10):
    img_tensor, label = train_set[i]
    writter.add_image('train_set', img_tensor, i)
writter.close()