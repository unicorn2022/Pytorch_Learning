from torchvision import transforms
from PIL import Image

# 1. 将PIL图片转换为Tensor
img_path = "data/train/ants/0013035.jpg"
img = Image.open(img_path)
img_tensor = transforms.ToTensor()(img)