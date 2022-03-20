import librosa
import os
import numpy as np


if __name__ == "__main__":
    time_list = []
    for i in range(16):
        time_list.append(0)
    max_len_file = None
    for root, dirs, files in os.walk("D:\\voicebank"):
        if len(dirs) == 0:
            if 'test' in root and 'clean' in root:
                max_len = 0
                for file in files:
                    signal, sr = librosa.load(os.path.join(root, file), sr=None)
                    signal_len = signal.shape[-1] / sr
                    signal_len = int(np.floor(signal_len))
                    time_list[signal_len] += 1
                    if max_len < signal_len:
                        max_len = signal_len
                        max_len_file = file
                print("max test audio length:", max_len+1, max_len_file)
                print(time_list)
            elif 'train' in root and 'clean' in root:
                time_list = []
                for i in range(16):
                    time_list.append(0)
                max_len = 0
                for file in files:
                    signal, sr = librosa.load(os.path.join(root, file), sr=None)
                    signal_len = signal.shape[-1] / sr
                    signal_len = int(np.floor(signal_len))
                    time_list[signal_len] += 1
                    if max_len < signal_len:
                        max_len = signal_len
                        max_len_file = file
                print("max train audio length:", max_len+1, max_len_file)
                print(time_list)
# max test audio length: 9 p232_023.wav
# [0, 260, 367, 167, 13, 10, 3, 1, 2, 1, 0, 0, 0, 0, 0, 0]
# max train audio length: 15 p243_035.wav
# [0, 1497, 5890, 2907, 847, 189, 100, 76, 26, 16, 10, 8, 4, 1, 0, 1]
