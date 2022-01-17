import librosa
import torch
import torchaudio

from mynet import conv_tasnet
from torch.utils.data import DataLoader


if __name__ == "__main__":
    net = conv_tasnet(num_speakers=2)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    loss_function = torch.nn.MSELoss()

    signal, sr = torchaudio.load("../audio/mix.wav")
    label1, sr = torchaudio.load("../audio/1.wav")
    label2, sr = torchaudio.load("../audio/2.wav")
    EPOCHS = 300
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        prediction_sum = net(signal)

        # torchaudio.save("pre2.wav", prediction_sum[:, 1, :].detach(), sr)
        label = torch.cat((label1, label2), dim=0)
        label = torch.unsqueeze(label, dim=0)
        loss = loss_function(label, prediction_sum)

        if best_loss >= loss.item():
            best_loss = loss.item()
        print(f"{epoch+1}: {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch + 1 == EPOCHS:
            torchaudio.save("pre1.wav", prediction_sum[:, 0, :].detach(), sr)
            torchaudio.save("pre2.wav", prediction_sum[:, 1, :].detach(), sr)
