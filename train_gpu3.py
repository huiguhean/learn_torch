import torch.optim
import torchvision
from matplotlib import transforms
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Linear, Flatten
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import Dataset
# import cv2
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from torchvision import transforms
# from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

# from model_save import Tudui
class MyData(Dataset):

    def __init__(self, root_dir, image_dir, label_dir, transform):
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.label_path = os.path.join(self.root_dir, self.label_dir)
        self.image_path = os.path.join(self.root_dir, self.image_dir)
        self.image_list = os.listdir(self.image_path)
        self.label_list = os.listdir(self.label_path)
        self.transform = transform
        # 因为label 和 Image文件名相同，进行一样的排序，可以保证取出的数据和label是一一对应的
        self.image_list.sort()
        self.label_list.sort()

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        label_name = self.label_list[idx]
        img_item_path = os.path.join(self.root_dir, self.image_dir, img_name)
        label_item_path = os.path.join(self.root_dir, self.label_dir, label_name)
        img = Image.open(img_item_path)

        with open(label_item_path, 'r') as f:
            label = int(f.readline())

        img = self.transform(img)
        return img, label

    def __len__(self):
        assert len(self.image_list) == len(self.label_list)
        return len(self.image_list)
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
            Linear(64,2)
        )
    def forward(self,x):
        x = self.modul1(x)
        return x

# train_data = torchvision.datasets.CIFAR10("./dataset",train=True,transform=torchvision.transforms.ToTensor())
# test_data = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())
if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    root_dir = "hymenoptera_data/train"
    image_ants = "ants_image"
    label_ants = "ants_label"
    ants_dataset = MyData(root_dir, image_ants, label_ants, transform)
    image_bees = "bees_image"
    label_bees = "bees_label"
    bees_dataset = MyData(root_dir, image_bees, label_bees, transform)
    # train_dataset = bees_dataset + ants_dataset
    train_dataset = ConcatDataset([ants_dataset, bees_dataset])
    # train_dataset = train_dataset.shuffle
# train_dataloader = DataLoader(train_data,batch_size=64)
# test_dataloader = DataLoader(test_data,batch_size=64)
#     train_dataloader = DataLoader(train_dataset, batch_size=35, num_workers=2)
#     train_dataloader = DataLoader(train_dataset, batch_size=35, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=35)
    test_dataloader = train_dataloader
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
            # data = list(data0.items())
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
        # print('正确率',total_accuracy/len(test_data))
        print('正确率', total_accuracy / len(train_dataset))
    torch.save(tudui,"tudui_gpu3.pth")