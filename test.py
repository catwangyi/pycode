import numpy as np
import torch
import torch.nn as nn

conv1 = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=3, stride=5)
transconv = nn.ConvTranspose1d(in_channels=4, out_channels=2, kernel_size=3, stride=5)

input = torch.rand(1, 2, 512)
outout = conv1(input)
output2 = transconv(outout)
print(a)