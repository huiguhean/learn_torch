import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_set = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())
test_loder = DataLoader(dataset=test_set,batch_size=64,shuffle=False,num_workers=0,drop_last=False)

writer = SummaryWriter("dataloader")
img,lab = test_set[0]
print(img.shape)
print(img)
step = 0
for data in test_loder:
    img,lab = data
    writer.add_images("imgl:{}".format(step),img,step)
    step = step+1
for i in range(30):
    img,lab = data
    writer.add_images("imgloder14",img,i)
writer.close()
