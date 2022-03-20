import torch
from torch.nn.modules import Module
import torch.nn as nn


class AECNN_T(Module):
    def __init__(self):
        super(AECNN_T, self).__init__()
        self.encoder = Aecnn_Encoder()
        self.middle = self.encoder.down_sample_block(in_channel=256,
                                                     out_channel=256)
        self.decoer = Aecnn_Decoder()
        self.output = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=1, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        encoder_out, skip_values = self.encoder(x)
        middle_out = self.middle(encoder_out)
        decoder_out = self.decoer(middle_out, skip_values)
        enhanced = self.output(torch.cat((decoder_out, skip_values[0]), dim=1))
        return enhanced


class Aecnn_Decoder(Module):
    def __init__(self, channel_list=[512, 512, 256, 256, 256, 128, 128, 128]):
        super(Aecnn_Decoder, self).__init__()
        self.up_con1 = self.deconv_blocks(in_channel=256,
                                          out_channel=256,
                                          )
        self.up_con2 = self.deconv_blocks(in_channel=512,
                                          out_channel=256,
                                          )
        self.up_con3 = self.deconv_blocks(in_channel=512,
                                          out_channel=128,
                                          )
        self.up_con4 = self.deconv_blocks(in_channel=256,
                                          out_channel=128,
                                          )
        self.up_con5 = self.deconv_blocks(in_channel=256,
                                          out_channel=128,
                                          )
        self.up_con6 = self.deconv_blocks(in_channel=256,
                                          out_channel=64,
                                          )
        self.up_con7 = self.deconv_blocks(in_channel=128,
                                          out_channel=64,
                                          )
        self.up_con8 = self.deconv_blocks(in_channel=128,
                                          out_channel=64)

    def forward(self, middle_out, skip_values):
        d1 = self.up_con1(middle_out)
        d2 = self.up_con2(torch.cat((d1, skip_values[7]), dim=1))
        d3 = self.up_con3(torch.cat((d2, skip_values[6]), dim=1))
        d4 = self.up_con4(torch.cat((d3, skip_values[5]), dim=1))
        d5 = self.up_con5(torch.cat((d4, skip_values[4]), dim=1))
        d6 = self.up_con6(torch.cat((d5, skip_values[3]), dim=1))
        d7 = self.up_con7(torch.cat((d6, skip_values[2]), dim=1))
        d8 = self.up_con8(torch.cat((d7, skip_values[1]), dim=1))
        return d8


    def deconv_blocks(self, in_channel, out_channel, kernel=14, stride=2, padding=6):
        block = nn.Sequential(
            nn.ConvTranspose1d(in_channels=in_channel,
                               out_channels=out_channel,
                               kernel_size=kernel,
                               stride=stride,
                               padding=padding
                               ),
            nn.BatchNorm1d(out_channel),
            nn.ReLU()
        )
        return block


class Aecnn_Encoder(Module):
    def __init__(self, channels_list=[1, 64, 64, 64, 128, 128, 128, 256, 256]):
        super(Aecnn_Encoder, self).__init__()
        self.down1 = self.down_sample_block(in_channel=channels_list[0],
                                            out_channel=channels_list[1],
                                            kernel=1,
                                            padding=0,
                                            stride=1)
        self.down2 = self.down_sample_block(in_channel=channels_list[1],
                                            out_channel=channels_list[2],
                                            )
        self.down3 = self.down_sample_block(in_channel=channels_list[2],
                                            out_channel=channels_list[3])
        self.down4 = self.down_sample_block(in_channel=channels_list[3],
                                            out_channel=channels_list[4])
        self.down5 = self.down_sample_block(in_channel=channels_list[4],
                                            out_channel=channels_list[5])
        self.down6 = self.down_sample_block(in_channel=channels_list[5],
                                            out_channel=channels_list[6])
        self.down7 = self.down_sample_block(in_channel=channels_list[6],
                                            out_channel=channels_list[7])
        self.down8 = self.down_sample_block(in_channel=channels_list[7],
                                            out_channel=channels_list[8])

    def forward(self, x):
        f1 = self.down1(x)
        f2 = self.down2(f1)
        f3 = self.down3(f2)
        f3 = nn.Dropout(p=0.2)(f3)
        f4 = self.down4(f3)
        f5 = self.down5(f4)
        f6 = self.down6(f5)
        f6 = nn.Dropout(p=0.2)(f6)
        f7 = self.down7(f6)
        f8 = self.down8(f7)
        return f8, [f1, f2, f3, f4, f5, f6, f7, f8]

    def down_sample_block(self, in_channel, out_channel, kernel=14, padding=6, stride=2):
        block = nn.Sequential(
            nn.Conv1d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=kernel,
                      padding=padding,
                      stride=stride
                      ),
            nn.BatchNorm1d(out_channel),
            nn.PReLU(),
        )
        return block


if __name__ =="__main__":
    net = AECNN_T()
    input = torch.ones((2, 2048, 1))
    input = input.permute(0, 2, 1)
    pred = net(input)

