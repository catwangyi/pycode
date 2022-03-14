import torchaudio
import soundfile
import torch
import librosa
import os
import torch.nn.functional as F
import numpy as np


if __name__ == "__main__":
    for root, dirs, files in os.walk("D:\\voicebank\\test\\clean_testset_wav"):
        for file in files:
            input_sig, sr = librosa.load(root + "\\" + file, sr=16000)
            x = np.floor(np.log2(sr * 5)) + 1
            needed_len = int(2 ** x)
            if input_sig.shape[-1] <= needed_len:
                rest_len = needed_len - input_sig.shape[-1]
                sig = np.pad(array=input_sig, pad_width=(0, rest_len), mode="constant", constant_values=0)
            else:
                start_idx = int((input_sig.shape[-1] - needed_len) / 2)
                sig = input_sig[:, start_idx:start_idx + needed_len]
            soundfile.write("audio\\" + file, data=sig, samplerate=sr)


    # for root, dirs, files in os.walk("D:\\voicebank\\test\\noisy_testset_wav"):
    #     for file in files:
    #         input_sig, sr = torchaudio.load(root + "\\" + file)
    #         x = np.floor(np.log2(sr * 5)) + 1
    #         needed_len = int(2 ** x)
    #         if input_sig.shape[-1] <= needed_len:
    #             rest_len = needed_len - input_sig.shape[-1]
    #             sig = F.pad(input=input_sig, pad=(0, rest_len), mode="constant", value=0)
    #         else:
    #             start_idx = int((input_sig.shape[-1] - needed_len) / 2)
    #             sig = input_sig[:, start_idx:start_idx + needed_len]
    #         torchaudio.save("audio\\" + file, sig, sr)


    # net = torch.load("best_model.pth")
    # with torch.no_grad():
    #     net.eval()
    #     input_sig, sr = torchaudio.load("p232_024.wav")
    #     x = np.floor(np.log2(sr * 5)) + 1
    #     needed_len = int(2 ** x)
    #     if input_sig.shape[1] <= needed_len:
    #         rest_len = needed_len - input_sig.shape[1]
    #         noisy_sig = F.pad(input=input_sig, pad=(0, rest_len), mode="constant", value=0)
    #     else:
    #         start_idx = int((input_sig.shape[1] - needed_len) / 2)
    #         noisy_sig = input_sig[:, start_idx:start_idx + needed_len]
    #     input = torch.unsqueeze(noisy_sig, dim=0)
    #     pred = net(input)
    #     pred = torch.reshape(pred, (pred.shape[-1], -1))
    #     soundfile.write("pred_audio.wav", pred.detach().numpy(), sr)
