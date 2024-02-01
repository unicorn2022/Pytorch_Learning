[TOC]

# 一、数据加载：Dataset & Dataloader

- **Dataset**：提供一种方式，获取`data`及其`label`

  - 如何获取单个`data`及其`label`
  - 计算总共有多少个`data`

  ```python
  from torch.utils.data import Dataset
  class MyDataset(Dataset):
      def __init__(self, data_path):
          ...
  
      def __len__(self):
          ...
      
      def __getitem__(self, index):
          ...
  ```

- **Dataloader**：将数据打包，提供给网络使用

<img src="AssetMarkdown/image-20240127232050876.png" alt="image-20240127232050876" style="zoom:80%;" />

# 二、Tensorboard

Python生成日志文件：

- 坐标轴：观察网络的`loss`

  ```python
  from torch.utils.tensorboard import SummaryWriter
  
  writer = SummaryWriter(
      log_dir='logs' # 存储路径, 建议每次训练使用不同的文件夹, 以免覆盖之前的训练结果
  )
  
  # 坐标轴
  for i in range(100):
      writer.add_scalar(
          tag='y=x^2',        # 坐标轴名称
          scalar_value=i**2,  # Y轴值
          global_step=i       # X轴值
      )
  
  writer.close()
  ```

- 图像：观察网络的图像输入or输出

  ```python
  # 图片
  img_path = 'data/train/ants/0013035.jpg'
  img_PIL = Image.open(img_path)
  img_array = np.array(img_PIL)
  writer.add_image(
      tag='test',             # 图片名称
      img_tensor=img_array,   # 图片数据
      global_step=0,          # 第几张图片
      dataformats='HWC'       # 图片数据格式(HWC:高度、宽度、通道数)
  )
  ```

使用`Tensorboard`打开日志文件：

```bash
tensorboard --logdir=logs --port=6007
```

查看port（如`6007`）被哪些进程占用：（可以得到进程PID）

```bash
netstat -ano | findstr 6007
```

查看PID（如`20288`）对应的进程

```bash
tasklist | findstr 20288
```

# 三、Transforms

<img src="AssetMarkdown/image-20240129232207604.png" alt="image-20240129232207604" style="zoom:80%;" />

## 3.1	将PIL图片转为Tensor类型

```python
from torchvision import transforms
from PIL import Image

# 1. 将PIL图片转换为Tensor
img_path = "data/train/ants/0013035.jpg"
img = Image.open(img_path)
img_tensor = transforms.ToTensor()(img)
```

- **Tensor**：包装了神经网络必须的一些属性
- **ToTensor()**：将PIL、ndarray格式的图片，转化为Tensor格式

## 3.2	常用Transforms

- **输入**：
  - **PIL**：`Image.open(path)`
  - **tensor**：`transforms.ToTensor()(PILImage)`
  - **ndarray**：`cv.imread(path)`
- **作用**
  - **将PIL、ndarray转化为Tensor**：`transforms.ToTensor()(PILImage)`
  - **将Tensor、ndarray转化为PIL**：`transforms.ToPILImage()(tensor)`
  - **按通道归一化Tensor图像**：`transforms.Normalize(mean, std)(tensor)`
  - **更改PIL图像大小**：`transforms.Resize(size)(PILImage)`
  - **组合多个作用**：`transforms.Compose([transform_1, transform_2...])`
  - **随机裁剪PIL图像**：`transforms.RandomCrop(size)(PILImage)`

# 四、torchvision数据集

```python
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
```

