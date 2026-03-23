import torch
from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def preprocess(img):

    img = Image.open(img).convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0)

    return img
