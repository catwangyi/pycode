from loss import freqLoss_From_TimeDomain
import torch
import numpy as np
from dataset_aecnn import AecnnDataset
from model import AECNN_T
from torch.utils.data import  DataLoader

DEVICE = 'cpu'


if __name__ == "__main__":
    dataset = AecnnDataset(audio_path="E:\dataset\\voicebank", device=DEVICE)
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    net = AECNN_T()
    net.to(DEVICE)
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
        optimizer.zero_grad()
        loss.backward(loss.clone().detach())
        optimizer.step()



