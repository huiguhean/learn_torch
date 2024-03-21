import torch
import torchvision
from model_save import *
model = torch.load("vgg16_save_method1.pth")
# print(model)

#方法2
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load('vgg16_save_method2.pth'))
#print(vgg16)

tudui = torch.load("vgg16_tudui_method1.pth")
print(tudui)