import torch.nn as nn

class Iris_Classification_Network(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        '''
        初始化: 定义网络的层级结构
        :param in_dim: 输入维度
        :param hidden_dim: 隐藏层维度
        :param out_dim: 输出维度
        '''
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        '''
        前向传播: 定义网络的前向传播过程
        :param data: 输入数据
        :return: 输出数据
        '''
        data = self.layer1(data)
        data = self.layer2(data)
        data = self.layer3(data)
        return data