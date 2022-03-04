from torch.utils.data import Dataset
import librosa


class seganDataset(Dataset):
    def __init__(self, datapath):
        super(seganDataset, self).__init__()
        librosa.load(datapath)

    def __getitem__(self, index):
        return 1

    def __len__(self):
        return 1