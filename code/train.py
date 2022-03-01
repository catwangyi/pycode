import librosa
import torch
import torchaudio

from mynet import conv_tasnet
from torch.utils.data import DataLoader


def sisinr(x, s, eps=1e-8):
    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(f"dimention mistmatch{x.shape} vs {s.shape}")
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(x_zm * s_zm, dim=-1, keepdim=True)
    t = t * s_zm / (l2norm(s_zm, keepdim=True) + eps)
    return 20 * torch.log10(eps + l2norm(t)) / (l2norm(x_zm - t) + eps)


if __name__ == "__main__":
    net = conv_tasnet(num_speakers=2)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-3)
    EPOCHS = 20

    loss = None
    signal, sr = librosa.load("../audio/mix.wav", sr=8000)
    signal = torch.from_numpy(signal).view(1, 1, -1)
    for epoch in range(EPOCHS):
        prediction = net(signal)
        a = 1
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
