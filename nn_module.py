from d2l import torch
from torch import nn


class gdw(nn.Module):
    def __init__(self):
        super(gdw,self).__init__()
    def forward(selfself,input):
        output = input+1
        return output
tudui = gdw()
x = torch.tensor(1.0)
output = tudui(x)
print(output)