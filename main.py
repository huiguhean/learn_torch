# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
from IPython import display

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    mnist_train = torchvision.datasets.FashionMNIST(root='D:\d2l-zh-2.0.0\pytorch\data', train=True, download=True,
                                                    transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='D:\d2l-zh-2.0.0\pytorch\data', train=False, download=True,
                                                   transform=transforms.ToTensor())
#数据集大小
print(type(mnist_train))
print(len(mnist_train), len(mnist_test))
# 本函数已保存在d2lzh_pytorch包中方便以后使用
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
# 本函数已保存在d2lzh_pytorch包中方便以后使用
def show_fashion_mnist(images, labels):
    display.set_matplotlib_formats('svg')#用矢量图进行展示
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        print(img.shape)
        print(img)
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()
X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
