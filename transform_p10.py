from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("images/2012.png")
print(img)
#totensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("totensor",img_tensor)

#归一化normalization
trans_norm = transforms.Normalize([2,1,3],[0.5,0.5,0.5])
img_normalize = trans_norm(img_tensor)
writer.add_image("normal",img_normalize,3)
#resize需要pil类的img不需要tensor类的img
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)#这里用img是pil类型，不是tensor类型
img_resize = trans_totensor(img_resize)#把pil类型转tensor好用writer
writer.add_image("Resize",img_resize,0)
#compose是吧多个tansforms链接在一起执行
#randomcrop随机裁剪大小
trans_random = transforms.RandomCrop((200,200))
trans_compose_2 = transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("randomCrop",img_crop,i)
writer.close()
#关注输入输出类型，关注官方文档
