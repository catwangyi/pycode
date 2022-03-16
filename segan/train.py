import torch.optim
from model import Generator, Discriminator
import torchaudio
from WaveUnet.dataset import MyDataset
from torch.utils.data import DataLoader
DATASET_PATH = "D:\\voicebank"


if __name__ == "__main__":

    dataset = MyDataset(DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    generator = Generator()
    discriminator = Discriminator()
    EPOCHS = 50
    g_optimizer = torch.optim.RMSprop(generator.parameters(), lr=1e-4)
    d_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=1e-4)
    sr = 16000
    best_d_score = float('inf')
    best_g_score = float('inf')
    for epoch in range(EPOCHS):
        for speech_noisy, speech_clean in dataloader:
            # discriminator training
            discriminator.zero_grad()
            d_input = torch.cat((speech_clean, speech_noisy), dim=1)
            # d_input : (2, 16384)
            d_output = discriminator(d_input)
            d_loss_1 = torch.mean((d_output - 1) ** 2) / 2

            generator_out = generator(speech_noisy)
            # 将带噪语音和增强后语音一起输入
            d_out = discriminator(torch.cat((generator_out, speech_noisy), dim=1))
            d_loss_2 = torch.mean(d_out ** 2) / 2

            d_loss = d_loss_1 + d_loss_2
            d_loss.backward()
            d_optimizer.step()

            # generator training
            generator.zero_grad()
            generator_out = generator(speech_noisy)

            d_input = torch.cat((generator_out, speech_noisy), dim=1)
            g_loss_1 = torch.mean((discriminator(d_input) - 1) ** 2) / 2
            g_loss_2 = 100 * torch.mean(torch.abs(torch.add(generator_out, torch.neg(speech_clean))))
            g_loss = g_loss_1 + g_loss_2
            g_loss.backward()
            g_optimizer.step()
            print(f"d_loss:{d_loss.item()}, g_loss:{g_loss.item()}")
            if d_loss.item() < best_d_score:
                best_d_score = d_loss.ietm()
                torch.save(discriminator.state_dict(), "best_d_net.pth")
                print("save discriminator")
            if g_loss.item() < best_g_score:
                best_g_score = g_loss.ietm()
                print("save generator")
                torch.save(generator.state_dict(), "best_g_net.pth")




