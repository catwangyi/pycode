import torch
from torch.utils.data import Dataset
import os
import torchaudio
import librosa
from util.utils import padsignal, audio_to_frame
import numpy as np
from loss import freqLoss_From_TimeDomain
from model import AECNN_T


class AecnnDataset(Dataset):
    def __init__(self, audio_path, device, hop_len, is_training=True):
        super(AecnnDataset, self).__init__()
        self.state = None
        self.device = device
        self.hop_len = hop_len
        self.clean_list = []
        self.noisy_list = []
        self.clean_folder = None
        self.noisy_folder = None
        if is_training:
            self.state = "train"
            self.path = os.path.join(audio_path, self.state)
        else:
            self.state = "test"
            self.path = os.path.join(audio_path, self.state)

        for root, dirs, files in os.walk(self.path):
            if len(files) != 0:
                if "clean" in root:
                    if self.clean_folder is None:
                        self.clean_folder = os.path.basename(root)
                    self.clean_list = [file for file in files]
                elif "noisy" in root:
                    if self.noisy_folder is None:
                        self.noisy_folder = os.path.basename(root)
                    self.noisy_list = [file for file in files]
        self.clean_list = sorted(self.clean_list)
        self.noisy_list = sorted(self.noisy_list)

    def __getitem__(self, index):
        clean_path = self.clean_list[index]
        noisy_path = self.noisy_list[index]

        label, sr = torchaudio.load(os.path.join(self.path, self.clean_folder, clean_path))
        noisy_sig, _ = torchaudio.load(os.path.join(self.path, self.noisy_folder, noisy_path))

        needed_len = np.ceil(((noisy_sig.shape[-1])-2048) / self.hop_len) * self.hop_len + 2048
        # needed_len : 114688.0
        noisy_sig = padsignal(input_sig=noisy_sig, needed_len=int(needed_len))
        label = padsignal(input_sig=label, needed_len=int(needed_len))

        noisy_frames = audio_to_frame(audio=noisy_sig,
                                      frame_size=2048,
                                      window=torch.ones(2048),
                                      hop_len=self.hop_len,
                                      device=self.device)

        spec_func = torchaudio.transforms.Spectrogram(n_fft=512,
                                                      hop_length=256,
                                                      onesided=False,
                                                      power=None,
                                                      window_fn=torch.hamming_window,
                                                      center=False,
                                                      return_complex=True
                                                      )
        label_spec = spec_func(label)
        return noisy_frames, label_spec

    def __len__(self):
        return len(self.noisy_list)


if __name__ == "__main__":
    dataset = AecnnDataset(audio_path="E:\dataset\\voicebank", device='cpu')
    x, y = dataset[0]
    # net = AECNN_T()
    # optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    # pred = None
    # for noisy, label in dataset:
    #     pred = torch.empty_like(noisy)
    #     for frame_idx in range(noisy.shape[-1]):
    #         pred[:, frame_idx] = net(noisy[:, frame_idx])
    #     loss = freqLoss_From_TimeDomain(enhanced_out=noisy, clean_spec=label, need_mean=True)
    #     print(loss.item())
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
