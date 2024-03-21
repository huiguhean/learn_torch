import torch
import torchvision.transforms
from PIL import Image
from torch import nn, device
from torch.nn import Sequential, MaxPool2d, Conv2d, Linear, Flatten
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

path = "images/cat.png"
img = Image.open(path).convert('RGB')

transfrom = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                            torchvision.transforms.ToTensor()])


img = transfrom(img)
img = torch.reshape(img,(1,3,32,32))

text_labels = ['飞机', 'automobile', '鸟', '猫', '鹿',
               '狗', '青蛙', '马', '船', '卡车']


tudui = torch.load("tudui_gup.pth",map_location=device('cpu'))
tudui.eval()
with torch.no_grad():
    output = tudui(img)

print(output.argmax(1).item())
print(text_labels[output.argmax(1).item()])

