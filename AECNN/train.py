import torch
import numpy as np
from dataset_aecnn import AecnnDataset
from model import AECNN_T
from torch.utils.data import  DataLoader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
from loss import freqLoss_From_TimeDomain


if __name__ == "__main__":
    dataset = AecnnDataset(audio_path="E:\dataset\\voicebank", device=DEVICE, hop_len=256)
    net = AECNN_T()
    net.to(DEVICE)
    best_loss = float('inf')
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    for noisy, label in dataset:
        noisy = noisy.to(DEVICE)
        label = label.to(DEVICE)
        pred = torch.empty_like(noisy)
        for frame_idx in range(noisy.shape[-1]):
            input = torch.reshape(noisy[:, frame_idx], (1, 1, 2048))
            pred[:, frame_idx] = torch.squeeze(net(input), dim=0)
            # pred :nan
        loss = freqLoss_From_TimeDomain(enhanced_out=pred, clean_spec=label, need_mean=True, device=DEVICE)
        print(loss.item())
        if loss.item() < best_loss:
          best_loss = loss.item()
          torch.save(net.state_dict(), "best_model.pth")
          print('save model')
        optimizer.zero_grad()
        loss.backward(loss.clone().detach())
        optimizer.step()
