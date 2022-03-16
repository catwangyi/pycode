import torch
import torchaudio
import os
import numpy as np
import torch.nn.functional as F
from model import WaveUnet


if __name__ == "__main__":
    in_channel_list = [1, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288]
    out_channel_list = [24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 288]
    net = WaveUnet(in_channel_list, out_channel_list)
    net.load_state_dict(state_dict=torch.load("best_model.pth", map_location='cpu'))
    with torch.no_grad():
        net.eval()
        for root, dirs, files in os.walk("D:\\voicebank\\test\\noisy_testset_wav"):
            if len(dirs) == 0:
                for file in files:
                    input_sig, sr = torchaudio.load(root + "\\" + file)
                    # x = np.floor(np.log2(sr * 10)) + 1
                    # needed_len = int(2 ** x)
                    # if input_sig.shape[1] <= needed_len:
                    #     rest_len = needed_len - input_sig.shape[1]
                    #     input_sig = F.pad(input=input_sig, pad=(0, rest_len), mode="constant", value=0)
                    # else:
                    #     start_idx = int((input_sig.shape[1] - needed_len) / 2)
                    #     input_sig = input_sig[:, start_idx:start_idx + needed_len]
                    pred = net(torch.unsqueeze(input_sig, dim=0))
                    pred = torch.squeeze(pred, dim=0)
                    torchaudio.save("D:\\voicebank\pred" + "\\pred_" + file, pred, sr, bits_per_sample=16)
                    # print("save:{}".format(file))