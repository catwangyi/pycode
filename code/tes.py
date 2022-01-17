import numpy
import torch
import torch.nn as nn
from torchsummary import summary

class tset(nn.Module):
    def __init__(self):
        super(tset, self).__init__()
        self.conv = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,
                              groups=2,
                              dilation=1)

    def forward(self, input):
        out = self.conv(input)
        print("---------{}".format(self.conv.weight.data.size()))
        return out


if __name__ == "__main__":
    net = tset()
    summary(net, input_size=(512, 28))
