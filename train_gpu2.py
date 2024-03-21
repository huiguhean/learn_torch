import torch.optim
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Linear, Flatten
from torch.utils.data import DataLoader
# from model_save import Tudui
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

train_data = torchvision.datasets.CIFAR10("./dataset",train=True,transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

device = torch.device("cuda:0")
#创建网络模型
tudui = Tudui()
tudui.to(device)
#损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
#优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(tudui.parameters(),lr=learning_rate)


train_step = 0
test_step = 0
epoch = 50

#训练开始步骤
for i in range(epoch):
    for data in train_dataloader:
        imgs,target = data
        imgs = imgs.to(device)
        target = target.to(device)
        output = tudui(imgs)
        loss = loss_fn(output,target)

        #优化器模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_step = train_step + 1
        # if train_step%100==0:
        #     print("训练次数:{},Loss:{}".format(train_step,loss.item()))

    #测试步骤
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,target = data
            imgs = imgs.to(device)
            target = target.to(device)
            output = tudui(imgs)
            loss = loss_fn(output, target)
            total_test_loss = total_test_loss + loss.item()
            accury = (output.argmax(1)==target).sum()
            total_accuracy = total_accuracy + accury

    print("{}次的测试集loss:{}".format(i,total_test_loss))
    print('正确率',total_accuracy/len(test_data))
torch.save(tudui,"tudui_gpu.pth")