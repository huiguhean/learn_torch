
import torchvision
import torchvision.transforms as transforms
daraset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR10(root="./dataset",train=True,transform=daraset_transform,download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=daraset_transform,download=True)

img ,lab= test_set[1]
# print(img)
to_pil_img = transforms.ToPILImage()
imgshow=to_pil_img(img)
imgshow.show()
