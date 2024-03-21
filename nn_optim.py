import torch.optim
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_set = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(test_set,batch_size=1)
# a,b = test_set[0]
# print(a.shape)
# for data in dataloader:
#     c,d = data
#     print(c.shape)
#     break
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

loss = nn.CrossEntropyLoss()
tudui = Tudui()
optim = torch.optim.SGD(tudui.parameters(),lr=0.01)

for epoch in range(20):
    runingloss = 0.0
    for data in dataloader:
        imgs,target = data
        outputs = tudui(imgs)
        result_loss = loss(outputs,target)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        runingloss = runingloss + result_loss
    print(runingloss )

# for data in dataloader:
