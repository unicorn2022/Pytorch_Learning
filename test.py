import torch

# 输出torch的版本
print("torch\t版本号:\t\t", torch.__version__)

# 输出cuda是否可用
print("cuda\t是否可用:\t", torch.cuda.is_available())