import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module


class WaveUnet(Module):
    def __init__(self, in_channel_list, out_channel_list):
        super(WaveUnet, self).__init__()
        self.encoder = Encoder(in_channel_list[:-1], out_channel_list[:-1])
        self.cat_tensors = []
        self.middle = nn.Sequential(
            nn.Conv1d(in_channels=out_channel_list[-1],
                      out_channels=out_channel_list[-1],
                      kernel_size=15,
                      padding=7,
                      stride=1),
            nn.BatchNorm1d(out_channel_list[-1]),
            nn.LeakyReLU(0.1)
        )
        decoder_in_list = [i+j for i, j in zip(in_channel_list[1:], out_channel_list[1:])]
        decoder_out = out_channel_list[::-1]
        self.decoder = Decoder(decoder_in_list[::-1], decoder_out[1:])
        self.last_layer = nn.Sequential(
                    nn.Conv1d(in_channels=in_channel_list[0] + out_channel_list[0], out_channels=1, kernel_size=1, stride=1, dilation=1),
                    nn.Tanh()
                )

    def forward(self, x):
        origin = x
        x, self.cat_tensors = self.encoder(x)
        x = self.middle(x)
        # self.cat_tensors.append(x)
        x = self.decoder(x, self.cat_tensors)
        return self.last_layer(torch.cat((origin, x), dim=1))


class Decoder(Module):
    def __init__(self, in_channel_list, out_channel_list):
        super(Decoder, self).__init__()
        self.model_list = nn.ModuleList()
        for i in range(len(in_channel_list)):
            self.model_list.append(UpConv(in_channel_list[i], out_channel_list[i]))

    def forward(self, x, cat_tensors):
        for layer, cat_tensor in zip(self.model_list, cat_tensors):
            x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=True)
            if x.shape[-1] == cat_tensor.shape[-1] - 1:
                x = F.pad(x, (0, 1), mode="reflect")
            elif x.shape[-1] == cat_tensor.shape[-1] + 1:
                x = x[:, :, :-1]
            x = torch.cat((x, cat_tensor), dim=1)
            x = layer(x)
        return x


class Encoder(Module):
    def __init__(self, in_channel_list, out_channel_list):
        super(Encoder, self).__init__()
        self.model_list = nn.ModuleList()
        self.feature_list = []
        for i in range(len(in_channel_list)):
            self.model_list.append(DownConv(in_channel_list[i], out_channel_list[i]))

    def forward(self, x):
        self.feature_list = []
        for layers in self.model_list:
            x = layers(x)
            self.feature_list.append(x)
            # if x.shape[-1] % 2:
                # x = F.pad(x, (0, 2), "constant", 0)
            x = x[:, :, ::2]
        return x, self.feature_list[::-1]


class DownConv(Module):
    def __init__(self, in_channel, out_channel):
        super(DownConv, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=15, padding=7, stride=1, dilation=1),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.seq(x)


class UpConv(Module):
    def __init__(self, in_channel, out_channel):
        super(UpConv, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=out_channel, padding=2, stride=1, kernel_size=5),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.seq(x)


if __name__ == "__main__":
    in_channel_list = [1, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288]
    out_channel_list = [24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 288]
    net = WaveUnet(in_channel_list, out_channel_list)
    print(net)


