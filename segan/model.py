import torch
import torch.nn as nn
from torchsummary import summary


class Generator(nn.Module):
    # Generator用encoder对带噪语音进行encode，
    # 再将其和随机变量z进行拼接，送入decoder
    def __init__(self):
        super(Generator, self).__init__()
        # encoder
        self.encoder_1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=32, stride=2, padding=15)
        self.encoder_1_nl = nn.PReLU()
        self.encoder_2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=32, stride=2, padding=15)
        self.encoder_2_nl = nn.PReLU()
        self.encoder_3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=32, stride=2, padding=15)
        self.encoder_3_nl = nn.PReLU()
        self.encoder_4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=32, stride=2, padding=15)
        self.encoder_4_nl = nn.PReLU()
        self.encoder_5 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=32, stride=2, padding=15)
        self.encoder_5_nl = nn.PReLU()
        self.encoder_6 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=32, stride=2, padding=15)
        self.encoder_6_nl = nn.PReLU()
        self.encoder_7 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=32, stride=2, padding=15)
        self.encoder_7_nl = nn.PReLU()
        self.encoder_8 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=32, stride=2, padding=15)
        self.encoder_8_nl = nn.PReLU()
        self.encoder_9 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=32, stride=2, padding=15)
        self.encoder_9_nl = nn.PReLU()
        self.encoder_10 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=32, stride=2, padding=15)
        self.encoder_10_nl = nn.PReLU()
        self.encoder_11 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=32, stride=2, padding=15)
        self.encoder_11_nl = nn.PReLU()

        # decoder
        self.decoder11 = nn.ConvTranspose1d(in_channels=1024 * 2, out_channels=512, kernel_size=32, stride=2,padding=15)
        self.decoder11_nl = nn.PReLU()
        self.decoder10 = nn.ConvTranspose1d(in_channels=512*2, out_channels=256, kernel_size=32, stride=2, padding=15)
        self.decoder10_nl = nn.PReLU()
        self.decoder9 = nn.ConvTranspose1d(in_channels=256*2, out_channels=256, kernel_size=32, stride=2, padding=15)
        self.decoder9_nl = nn.PReLU()
        self.decoder8 = nn.ConvTranspose1d(in_channels=256*2, out_channels=128, kernel_size=32, stride=2, padding=15)
        self.decoder8_nl = nn.PReLU()
        self.decoder7 = nn.ConvTranspose1d(in_channels=128*2, out_channels=128, kernel_size=32, stride=2, padding=15)
        self.decoder7_nl = nn.PReLU()
        self.decoder6 = nn.ConvTranspose1d(in_channels=128*2, out_channels=64, kernel_size=32, stride=2, padding=15)
        self.decoder6_nl = nn.PReLU()
        self.decoder5 = nn.ConvTranspose1d(in_channels=64*2, out_channels=64, kernel_size=32, stride=2, padding=15)
        self.decoder5_nl = nn.PReLU()
        self.decoder4 = nn.ConvTranspose1d(in_channels=64*2, out_channels=32, kernel_size=32, stride=2, padding=15)
        self.decoder4_nl = nn.PReLU()
        self.decoder3 = nn.ConvTranspose1d(in_channels=32*2, out_channels=32, kernel_size=32, stride=2, padding=15)
        self.decoder3_nl = nn.PReLU()
        self.decoder2 = nn.ConvTranspose1d(in_channels=32*2, out_channels=16, kernel_size=32, stride=2, padding=15)
        self.decoder2_nl = nn.PReLU()
        self.decoder1 = nn.ConvTranspose1d(in_channels=16*2, out_channels=1, kernel_size=32, stride=2, padding=15)
        self.decoder1_nl = nn.PReLU()

    def forward(self, input):
        enout_1 = self.encoder_1_nl(self.encoder_1(input))
        enout_2 = self.encoder_2_nl(self.encoder_2(enout_1))
        enout_3 = self.encoder_3_nl(self.encoder_3(enout_2))
        enout_4 = self.encoder_4_nl(self.encoder_4(enout_3))
        enout_5 = self.encoder_5_nl(self.encoder_5(enout_4))
        enout_6 = self.encoder_6_nl(self.encoder_6(enout_5))
        enout_7 = self.encoder_7_nl(self.encoder_7(enout_6))
        enout_8 = self.encoder_8_nl(self.encoder_8(enout_7))
        enout_9 = self.encoder_9_nl(self.encoder_9(enout_8))
        enout_10 = self.encoder_10_nl(self.encoder_10(enout_9))
        enout_11 = self.encoder_11_nl(self.encoder_11(enout_10))

        z = torch.randn_like(enout_11)
        dec_input = torch.cat((enout_11, z), dim=1)
        decout_11 = self.decoder11_nl(self.decoder11(dec_input))
        dec_input = torch.cat((decout_11, enout_10), dim=1)
        decout_10 = self.decoder10_nl(self.decoder10(dec_input))
        dec_input = torch.cat((decout_10, enout_9), dim=1)
        decout_9 = self.decoder9_nl(self.decoder9(dec_input))
        dec_input = torch.cat((decout_9, enout_8), dim=1)
        decout_8 = self.decoder8_nl(self.decoder8(dec_input))
        dec_input = torch.cat((decout_8, enout_7), dim=1)
        decout_7 = self.decoder7_nl(self.decoder7(dec_input))
        dec_input = torch.cat((decout_7, enout_6), dim=1)
        decout_6 = self.decoder6_nl(self.decoder6(dec_input))
        dec_input = torch.cat((decout_6, enout_5), dim=1)
        decout_5 = self.decoder5_nl(self.decoder5(dec_input))
        dec_input = torch.cat((decout_5, enout_4), dim=1)
        decout_4 = self.decoder4_nl(self.decoder4(dec_input))
        dec_input = torch.cat((decout_4, enout_3), dim=1)
        decout_3 = self.decoder3_nl(self.decoder3(dec_input))
        dec_input = torch.cat((decout_3, enout_2), dim=1)
        decout_2 = self.decoder2_nl(self.decoder2(dec_input))
        dec_input = torch.cat((decout_2, enout_1), dim=1)
        decout_1 = self.decoder1_nl(self.decoder1(dec_input))

        return nn.Tanh()(decout_1)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=31, stride=2, padding=15)
        self.batch_norm1 = nn.BatchNorm1d(num_features=32)
        self.relu1 = nn.PReLU()

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=31, stride=2, padding=15)
        self.batch_norm2 = nn.BatchNorm1d(num_features=64)
        self.relu2 = nn.PReLU()

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=31, stride=2, padding=15)
        self.batch_norm3 = nn.BatchNorm1d(num_features=128)
        self.relu3 = nn.PReLU()

        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=31, stride=2, padding=15)
        self.batch_norm4 = nn.BatchNorm1d(num_features=256)
        self.relu4 = nn.PReLU()

        self.conv5 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=31, stride=2, padding=15)
        self.batch_norm5 = nn.BatchNorm1d(num_features=512)
        self.relu5 = nn.PReLU()

        self.conv6 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=31, stride=2, padding=15)
        self.batch_norm6 = nn.BatchNorm1d(num_features=1024)
        self.relu6 = nn.PReLU()

        self.conv7 = nn.Conv1d(in_channels=1024, out_channels=2048, kernel_size=31, stride=2, padding=15)
        self.batch_norm7 = nn.BatchNorm1d(num_features=2048)
        self.relu7 = nn.PReLU()

        self.conv8 = nn.Conv1d(in_channels=2048, out_channels=1, kernel_size=1, stride=1)
        self.batch_norm8 = nn.BatchNorm1d(num_features=1)
        self.relu8 = nn.PReLU()

        self.fc_layer = nn.Linear(in_features=128, out_features=1)
        self.output = nn.Sigmoid()

    def forward(self, input):
        out1 = self.relu1(self.batch_norm1(self.conv1(input)))
        out2 = self.relu2(self.batch_norm2(self.conv2(out1)))
        out3 = self.relu3(self.batch_norm3(self.conv3(out2)))
        out4 = self.relu4(self.batch_norm4(self.conv4(out3)))
        out5 = self.relu5(self.batch_norm5(self.conv5(out4)))
        out6 = self.relu6(self.batch_norm6(self.conv6(out5)))
        out7 = self.relu7(self.batch_norm7(self.conv7(out6)))
        out8 = self.relu8(self.batch_norm8(self.conv8(out7)))
        out = torch.squeeze(out8)
        out = self.output(self.fc_layer(out))
        return out


if __name__ == "__main__":
    # generator = Generator()
    # summary(model=generator, input_size=(1, 16384), batch_size=2, device="cpu")
    discriminator = Discriminator()
    summary(model=discriminator, input_size=(2, 16384), batch_size=2)



