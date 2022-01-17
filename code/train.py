import torch
from mynet import conv_tasnet
from torch.utils.data import DataLoader


if __name__ == "__main__":
    net = conv_tasnet(num_speakers=2)
