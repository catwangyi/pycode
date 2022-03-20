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
    def __init__(self, audio_path, device,  is_training=True):
        super(AecnnDataset, self).__init__()
        self.state = None
        self.device = device
        self.clean_list = []
        self.noisy_list = []
        if is_training:
            self.state = "train"
            self.path = os.path.join(audio_path, self.state)
        else:
            self.state = "test"
            self.path = os.path.join(audio_path, self.state)

        for root, dirs, files in os.walk(self.path):
            if len(files) != 0:
                if "clean" in root:
                    self.clean_list = [root + "\\" + file for file in files]
                elif "noisy" in root:
                    self.noisy_list = [root + "\\" + file for file in files]

    def __getitem__(self, index):
        clean_path = self.clean_list[index]
        noisy_path = self.noisy_list[index]

        label, sr = librosa.load(clean_path, sr=16000)
        noisy_sig, _ = librosa.load(noisy_path, sr=sr)

        needed_len = np.ceil((len(label)-2048)/256) * 256 + 2048
        # needed_len : 114688.0
        noisy_sig = padsignal(input_sig=noisy_sig, needed_len=int(needed_len))
        label = padsignal(input_sig=label, needed_len=int(needed_len))
        label = torch.from_numpy(label)

        noisy_sig = torch.from_numpy(noisy_sig)
        noisy_frames = audio_to_frame(audio=noisy_sig,
                                      frame_size=2048,
                                      window=torch.hann_window(2048),
                                      device=self.device)

        spec_func = torchaudio.transforms.Spectrogram(n_fft=512,
                                                      hop_length=256,
                                                      onesided=False,
                                                      power=None,
                                                      center=False,
                                                      return_complex=True
                                                      )
        label_spec = spec_func(label)
        return noisy_frames, label_spec

    def __len__(self):
        return len(self.noisy_list)


if __name__ == "__main__":
    dataset = AecnnDataset(audio_path="D:\\voicebank")
    net = AECNN_T()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    pred = None
    for noisy, label in dataset:
        pred = torch.empty_like(noisy)
        for frame_idx in range(noisy.shape[-1]):
            pred[:, frame_idx] = net(noisy[:, frame_idx])
        loss = freqLoss_From_TimeDomain(enhanced_out=noisy, clean_spec=label, need_mean=True)
        print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
