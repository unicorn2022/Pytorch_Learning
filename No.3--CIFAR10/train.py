import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from nn_CIFAR10 import MyModel

# 数据集
dataset_path = './dataset'
dataset_transform = transforms.ToTensor()
dataset_train = datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=dataset_transform)
dataset_test  = datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=dataset_transform)
print('训练集大小:', len(dataset_train))
print('测试集大小:', len(dataset_test))

# 利用 DataLoader 加载数据
dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=0)
dataloader_test  = DataLoader(dataset_test,  batch_size=64, shuffle=False, num_workers=0)

# 创建网络模型
model = MyModel()

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()

# 创建优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

# 设置训练参数
epoch = 10
total_train_step = 0

# 使用 tensorboard 记录训练过程
writer = SummaryWriter(log_dir='./log')

for i in range(epoch):
    print('-------第 {} 轮训练开始-------'.format(i+1))
    
    # 训练网络
    model.train()
    for data in dataloader_train:
        # 模型推理
        img, target = data
        output = model(img)
        loss = loss_fn(output, target)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录训练过程
        total_train_step += 1
        if total_train_step % 100 == 0:
            writer.add_scalar('train_step_loss', loss.item(), total_train_step)
            print('第 {} 轮训练, 第 {} 步, 损失值: {:.4f}'.format(i+1, total_train_step, loss.item()))
    
    # 测试网络
    model.eval()
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad(): # 测试时不需要计算梯度
        for data in dataloader_train:
            img, target = data
            output = model(img)
            # 计算损失
            loss = loss_fn(output, target)
            total_loss += loss.item()
            # 计算准确率
            accuracy = (output.argmax(dim=1) == target).sum()
            total_accuracy += accuracy.item()
    total_accuracy_rate = total_accuracy / len(dataset_train)
    writer.add_scalar('train_total_loss', total_loss, i+1)
    writer.add_scalar('train_total_accuracy', total_accuracy_rate, i+1)
    print('第 {} 轮训练结束, 总损失值: {:.4f}, 准确率: {:.4f}'.format(i+1, total_loss, total_accuracy_rate))

    # 保存网络
    torch.save(model, './results/model_{}.pth'.format(i+1))
    print('第 {} 轮模型保存成功'.format(i+1))

writer.close()