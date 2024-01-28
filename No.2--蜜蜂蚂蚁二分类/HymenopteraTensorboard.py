from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter(log_dir='logs')
# 坐标轴
for i in range(100):
    writer.add_scalar(
        tag='y=x^2',        # 坐标轴名称
        scalar_value=i**2,  # Y轴值
        global_step=i       # X轴值
    )

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


writer.close()