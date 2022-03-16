import torchaudio
import soundfile
import torch
import librosa
import os
import torch.nn.functional as F
import numpy as np


if __name__ == "__main__":
    for root, dirs, files in os.walk("D:\\voicebank"):
        if len(dirs) == 0:
            for file in files:
                input_sig, sr = librosa.load(root + "\\" + file, sr=16000)
                x = np.floor(np.log2(sr * 4))
                needed_len = 16384
                if input_sig.shape[-1] < needed_len:
                    rest_len = needed_len - input_sig.shape[-1]
                    sig = np.pad(array=input_sig, pad_width=(0, rest_len), mode="constant", constant_values=0)
                elif input_sig.shape[-1] > needed_len:
                    start_idx = 8192
                    sig = input_sig[start_idx:start_idx + needed_len]
                soundfile.write(root + "\\" + file, data=sig, samplerate=sr)


