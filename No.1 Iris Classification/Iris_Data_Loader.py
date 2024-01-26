from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import torch

# 数据加载类必须继承自 torch.utils.data.Dataset
class Iris_Data_Loader(Dataset):
    def __init__(self, data_path):
        '''
        初始化: 设置数据集存储路径, 加载数据集
        :param data_path: 数据集存储路径
        '''
        self.data_path = data_path
        assert os.path.exists(self.data_path), '数据集不存在'
        
        # 加载数据
        datafile = pd.read_csv(self.data_path, names=[0,1,2,3,4])
        iris_name_map = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2 }

        # 初始化数据集 (pandas类型)
        datafile[4] = datafile[4].map(iris_name_map)
        data = datafile.iloc[:, 0:4]
        label = datafile.iloc[:, 4:]

        # 数据集归一化(Z值化)
        data = (data - np.mean(data)) / np.std(data)

        # 数据类型: pandas => numpy => tensor
        self.data = torch.from_numpy(np.array(data, dtype=np.float32))
        self.label = torch.from_numpy(np.array(label, dtype=np.int64))
        self.data_num = len(label)
        
        # 保证数据为可迭代类型, 便于后续通过 index 获取
        self.data = list(self.data)
        self.label = list(self.label)
        print('当前数据集大小: {}'.format(self.data_num))

    def __len__(self):
        '''
        获取数据集大小
        '''
        return self.data_num
    
    def __getitem__(self, index):
        '''
        获取数据集中的数据
        :param index: 数据索引
        '''
        return self.data[index], self.label[index]