import torch.optim
import librosa
from model import Generator, Discriminator
import torchaudio


if __name__ == "__main__":
    generator = Generator()
    discriminator = Discriminator()
    EPOCHS = 200
    g_optimizer = torch.optim.RMSprop(generator.parameters(), lr=1e-4)
    d_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=1e-4)
    speech_clean, sr = librosa.load("clean.wav", sr=None)
    speech_noisy, _ = librosa.load("noisy.wav", sr=None)

    speech_clean = torch.unsqueeze(torch.as_tensor(speech_clean[10048:26432]), dim=0)
    speech_noisy = torch.unsqueeze(torch.as_tensor(speech_noisy[10048:26432]), dim=0)
    speech_clean = torch.unsqueeze(speech_clean, dim=0)
    speech_noisy = torch.unsqueeze(speech_noisy, dim=0)

    for epoch in range(EPOCHS):
        # discriminator training
        discriminator.zero_grad()
        d_input = torch.cat((speech_clean, speech_noisy), dim=1)
        # d_input : (2, 16384)
        d_output = discriminator(d_input)
        d_loss_1 = torch.mean((d_output-1)**2)/2

        generator_out = generator(speech_noisy)
        # 将带噪语音和增强后语音一起输入
        d_out = discriminator(torch.cat((generator_out, speech_noisy), dim=1))
        d_loss_2 = torch.mean(d_out**2)/2

        d_loss = d_loss_1 + d_loss_2
        d_loss.backward()
        d_optimizer.step()

        # generator training
        generator.zero_grad()
        generator_out = generator(speech_noisy)

        d_input = torch.cat((generator_out, speech_noisy), dim=1)
        g_loss_1 = torch.mean((discriminator(d_input)-1)**2)/2
        g_loss_2 = 100 * torch.mean(torch.abs(torch.add(generator_out, torch.neg(speech_clean))))
        g_loss = g_loss_1 + g_loss_2
        g_loss.backward()
        g_optimizer.step()
        print(f"d_loss:{d_loss.item()}, g_loss:{g_loss.item()}")
        if epoch == EPOCHS - 1:
            torchaudio.save("generator_output.wav", torch.squeeze(generator_out, dim=0).detach(), sr)


