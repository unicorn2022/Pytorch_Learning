import os
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim

from Iris_Dataset import Iris_Dataset
from Iris_Classification_Network import Iris_Classification_Network

# 定义计算环境
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 从文件中加载数据
custom_dataset = Iris_Dataset('iris_data/Iris_data.txt')

# 划分数据集: 训练集(7), 验证集(2), 测试集(1)
train_size = int(len(custom_dataset) * 0.7)
valid_size = int(len(custom_dataset) * 0.2)
test_size = len(custom_dataset) - train_size - valid_size
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [train_size, valid_size, test_size])

# 加载数据集
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
print('训练集大小: ', len(train_loader) * 16)
print('验证集大小: ', len(valid_loader))
print('测试集大小: ', len(test_loader))

# 定义推理函数
def infer(model, dataset, device):
    # 模型进入推理模式
    model.eval()

    # 网络推理
    acc_num = 0
    with torch.no_grad():
        for data in dataset:
            datas, label = data
            # outputs有2维: [batch_size][out_dim]
            outputs = model(datas.to(device))
            # 预测结果, torch.max()的返回值为: tuple(最大值, 最大值的索引)
            prefict_y = torch.max(outputs, dim=1)[1]
            # 与label进行比较, 记录正确预测数量
            acc_num += torch.eq(prefict_y, label.to(device)).sum().item()
    
    # 正确率
    acc = acc_num / len(dataset)
    return acc

# 定义训练函数
in_dim = 4
out_dim = 3
def main(lr = 0.005, epochs = 20):
    # 定义网络, 损失函数, 优化器
    model = Iris_Classification_Network(in_dim, 12, out_dim).to(device)
    model_params = [param for param in model.parameters() if param.requires_grad]
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_params, lr=lr)

    # 模型权重保存路径
    save_path = os.path.join(os.getcwd(), 'results')
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    
    # 开始训练
    for epoch in range(epochs):
        # 模型进入训练模式
        model.train()

        # 初始化正确率, 采样个数
        acc_num = torch.zeros(1).to(device)
        sample_num = 0
        train_acc = 0

        # 进行训练
        # 可视化进度条
        train_bar = tqdm(train_loader, file=sys.stdout, ncols=100)  
        for datas in train_bar:
            data, label = datas
            
            # 删除标签的最后一个维度, 防止多余封装
            label = label.squeeze(-1)
            
            # 采样个数
            sample_num += data.shape[0]
            
            # 梯度清零(初始化优化器): 消除历史记录对当前梯度的影响
            optimizer.zero_grad()
            
            # 网络推理
            outputs = model(data.to(device))
            pred_class = torch.max(outputs, dim=1)[1]
            acc_num = torch.eq(pred_class, label.to(device)).sum().item()
            
            # 计算损失, 并求导
            loss = loss_function(outputs, label.to(device))
            loss.backward()

            # 更新参数
            optimizer.step()

            # 输出训练信息
            train_acc = acc_num / sample_num
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        # 进行验证
        valid_acc = infer(model, valid_loader, device)
        print("train epoch[{}/{}] loss:{:.3f} 训练集准确率:{:.3f} 验证集准确率:{:.3f}".format(epoch + 1, epochs, loss, train_acc, valid_acc))

        # 保存模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, 'Iris_epoch_{}.pth'.format(epoch + 1)))
    print('训练完成!')

    # 测试模型
    test_acc = infer(model, test_loader, device)
    print('测试集正确率: ', test_acc)

if __name__ == '__main__':
    main(lr=0.005, epochs=20)