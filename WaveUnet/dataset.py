import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import os.path


class MyDataset(Dataset):
    def __init__(self, audio_path, is_training=True):
        super(MyDataset, self).__init__()
        self.state = None
        self.clean_list = []
        self.noisy_list = []
        if is_training:
            self.state = "train"
            self.path = os.path.join(audio_path, self.state)
        else:
            self.state = "test"
            self.path = os.path.join(audio_path, self.state)

        for root, dirs, files in os.walk(self.path):
            if 'txt' in dirs:
                dirs.remove('txt')
            if len(files) != 0:
                if "clean" in root:
                    self.clean_list = [root + "\\" + file for file in files]
                elif "noisy" in root:
                    self.noisy_list = [root + "\\" + file for file in files]

    def __getitem__(self, index):
        clean_path = self.clean_list[index]
        noisy_path = self.noisy_list[index]

        label, sr = torchaudio.load(clean_path)
        noisy_sig, _ = torchaudio.load(noisy_path)
        needed_len = sr * 5
        if label.shape[1] <= needed_len:
            rest_len = needed_len - label.shape[1]
            noisy_sig = F.pad(input=noisy_sig, pad=(0, rest_len), mode="constant", value=0)
            label = F.pad(input=label, pad=(0, rest_len), mode="constant", value=0)
        else:
            start_idx = int((label.shape[1] - needed_len)/2)
            noisy_sig = noisy_sig[:, start_idx:start_idx + needed_len]
            label = label[:, start_idx:start_idx + needed_len]
        # 如果不是2的倍数
        # if label.shape[1] % 2:
        #     noisy_sig = F.pad(input=noisy_sig, pad=(0, 1), mode="constant", value=0)
        #     label = F.pad(input=label, pad=(0, 1), mode="constant", value=0)
        return noisy_sig, label

    def __len__(self):
        return len(self.noisy_list)


if __name__ == "__main__":
    a = torch.Tensor(1, 1, 6)
    b = a[:, :, 1:4]
    print(b)
