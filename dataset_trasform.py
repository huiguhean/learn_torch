# import cv2
import numpy as np
import torchvision
# import cv2
from PIL import Image
from d2l import torch
from matplotlib import transforms, pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import matplotlib.image as img
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms
from PIL import Image
def transform_invert(self, img, show=False):
    # Tensor -> PIL.Image
    # 注意：img.shape = [3,32,32] cifar10中的一张图片，经过transform后的tensor格式

    if img.dim() == 3:  # single image # 3,32,32
        img = img.unsqueeze(0)  # 在第0维增加一个维度 1,3,32,32
    low = float(img.min())
    high = float(img.max())
    # img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))  # (img - low)/(high-low)
    grid = img.squeeze(0)  # 去除维度为1的维度
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    img = Image.fromarray(ndarr)
    if show:
        img.show()
    return img
daraset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR10(root="./dataset",train=True,transform=daraset_transform,download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=daraset_transform,download=True)
# train_set = torchvision.datasets.CIFAR10(root="./dataset",train=True,download=True)
# test_set = torchvision.datasets.CIFAR10(root="./dataset",train=False,download=True)
# a ,b= test_set[0]
# print(test_set.classes)
# print(a)
# print(test_set.classes[b])
# a.show()
# print(test_set[0])


writer = SummaryWriter("logs")
for i in range(11):
    img,target = test_set[i]
    writer.add_image("test_set",img,i)

writer.close()


# img ,b= test_set[0]
# print(img)
# to_pil_img = transforms.ToPILImage()
# imgshow=to_pil_img(img)
# imgshow.show()
