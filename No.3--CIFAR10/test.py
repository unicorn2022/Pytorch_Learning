import torch
import torchvision.transforms as transforms
from PIL import Image
from nn_CIFAR10 import MyModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 输入数据
image_path = "./imgs/dog.png"
image = Image.open(image_path).convert("RGB")

# 对输入数据进行变换, 使其符合模型输入要求
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])
image = transform(image)
image = torch.reshape(image, (1, 3, 32, 32))
image = image.to(device)

# 加载模型
model = torch.load("./results/model_10.pth", map_location=device)

# 预测
model.eval()
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)
    print(predicted.item())