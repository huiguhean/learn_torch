import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential

vgg16 = torchvision.models.vgg16(pretrained = False)
torch.save(vgg16,"vgg16_save_method1.pth")

#方法二
torch.save(vgg16.state_dict(),'vgg16_save_method2.pth')

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        self.modul1 = Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )
    def forward(self,x):
        x = self.modul1(x)
        return x

tudui = Tudui()
print(tudui)
torch.save(tudui,"vgg16_tudui_method1.pth")