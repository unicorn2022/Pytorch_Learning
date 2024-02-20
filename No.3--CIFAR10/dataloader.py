import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 测试数据集
test_data = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=torchvision.transforms.ToTensor())
if __name__ == '__main__':
    img, target = test_data[0]
    # print(img.shape)    # torch.Size([3, 32, 32])

# 测试数据集的加载器
test_loader = DataLoader(
    dataset=test_data,  # 数据集
    batch_size=64,      # 每个batch取多少个样本
    shuffle=True,       # 是否打乱
    num_workers=0,      # 读取的线程个数
    drop_last=True      # 是否丢弃最后一个batch
)
if __name__ == '__main__':
    writer = SummaryWriter('./logs')
    for epoch in range(2):
        step = 0
        for data in test_loader:
            imgs, targets = data
            # print(imgs.shape)   # torch.Size([batch_size, 3, 32, 32])
            writer.add_images('Epoch: {}'.format(epoch), imgs, step)
            step += 1
    writer.close()