import torch
import numpy as np
import torch.nn as nn
import torchsummary
from torch.autograd import Variable


class conv_tasnet(nn.Module):
    def __init__(self, num_speakers=2):
        super(conv_tasnet, self).__init__()
        self.encoder = nn.Conv1d(in_channels=1, out_channels=512, kernel_size=32, stride=16, bias=False)
        self.separater = separation(stacks=3, layers=8)
        self.decoder = nn.ConvTranspose1d(in_channels=512,
                                          out_channels=1,
                                          kernel_size=32,
                                          stride=16,
                                          bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):

        signal_length = input.size(2)
        batch_size = input.size(0)
        # rest 是指输入信号和输出信号长度之差，因为卷积和转置卷积长度并不完全一样
        rest = int(signal_length - np.floor((signal_length - 32)/16) * 16 - 32)

        if rest > 0:
            # 需要将input进行填充，填充为长度是32(也就是kernal)的整数倍的长度，此时转置卷积后的长度与填充后的长度保持一致
            need_legth = int(np.ceil(signal_length / 32) * 32 - signal_length)
            pad = Variable(torch.zeros(batch_size, 1, need_legth)).type(input.type())
            input = torch.cat([input, pad], 2)
        # print(rest)

        sound_feature = self.encoder(input)
        mask = self.separater(sound_feature)
        mask = self.sigmoid(mask)
        # mask : [2,1024,2002]
        mask = mask.view(mask.size(0), 2, 512, -1)
        masked_feature = mask * sound_feature

        masked_feature = masked_feature.view(mask.size(0)*mask.size(1), 512, -1)
        out = self.decoder(masked_feature)
        out = out.view(input.size(0), 2, -1)
        out = out[:, :, :-need_legth]
        return out


class newNormal(nn.Module):
    def __init__(self, num_features):
        super(newNormal, self).__init__()
        self.eps = 1e-8
        self.gains = nn.Parameter(torch.ones(1, num_features, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1))

    def forward(self, input):
        # input(batchsize, features, timeseq)
        batchsize = input.size(0)
        # 所有特征加起来 fearture:(batchsize, time)
        feature_sum = input.sum(1)
        # 按照时间顺序累加，每一点为当前点和之前点的累加
        cum_fearure_sum = torch.cumsum(feature_sum, dim=1)
        # 计算均值
        divid = np.arange(1, input.size(-1)+1)
        divid = torch.from_numpy(np.dot(divid, 512))

        mean = cum_fearure_sum / divid

        # 计算方差
        var_feature_sum = input.pow(2).sum(1)
        cum_var_sum = torch.cumsum(var_feature_sum, dim=1)
        # 平方的均值
        mean_pow = cum_var_sum / divid
        var = mean_pow - torch.pow(input=mean, exponent=2)

        mean = torch.unsqueeze(mean, dim=1)
        mean = mean.expand_as(input)
        var = torch.unsqueeze(var, dim=1)
        var = var.expand_as(input)

        x = (input - mean) / torch.sqrt(var + self.eps)
        out = x * self.gains.expand_as(input).type(input.type()) + self.bias.expand_as(input).type(input.type())
        return out


class separation(nn.Module):
    def __init__(self, stacks, layers):
        super(separation, self).__init__()
        self.stacks = stacks
        self.layers = layers
        self.layer_norm = newNormal(num_features=512)
        self.bottle_neck = nn.Conv1d(in_channels=512, out_channels=128, kernel_size=1)
        self.tcn_layers = nn.Sequential()
        for s in range(stacks):
            for l in range(layers):
                self.tcn_layers.add_module(name='dconv{}'.format((s+1)*l),
                                           module=depth_conv(in_channel=128,
                                                             hidden_channel=128*4,
                                                             out_channel=128,
                                                             dilation=2 ** l,
                                                             padding=2 ** l
                                                             ))
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv1d(in_channels=128, out_channels=2*512, kernel_size=1)
                                    )

    def forward(self, input):
        out = self.layer_norm(input)
        out = self.bottle_neck(out)
        out = self.tcn_layers(out)
        out = self.output(out)
        return out


class depth_conv(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel, dilation=1, padding = 1):
        super(depth_conv, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=hidden_channel, kernel_size=1)
        self.dconv = nn.Conv1d(in_channels=hidden_channel,
                               out_channels=hidden_channel,
                               kernel_size=3,
                               dilation=dilation,
                               groups=hidden_channel,
                               padding=padding)
        self.non_linear1 = nn.PReLU()
        self.non_linear2 = nn.PReLU()
        self.normal1 = newNormal(num_features=hidden_channel)
        self.normal2 = newNormal(num_features=hidden_channel)
        self.output = nn.Conv1d(in_channels=hidden_channel, out_channels=out_channel, kernel_size=1)

    def forward(self, input):
        out = self.normal1(self.non_linear1(self.conv1(input)))
        out = self.output(self.normal2(self.non_linear2(self.dconv(out))))
        return out


if __name__ == "__main__":
    net = conv_tasnet(2)
    torchsummary.summary(net, (1, 16375))
