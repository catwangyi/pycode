import torch
import torchaudio
import os


if __name__ == "__main__":
    net = torch.load("best_model.pth")
    with torch.no_grad():
        net.eval()
        for root, dirs, files in os.walk("audio\\noisy"):
            for file in files:
                input_sig, sr = torchaudio.load(root+"\\"+file)
                input_sig = torch.unsqueeze(input_sig, dim=0)
                pred = net(input_sig)
                torchaudio.save("audio\pred"+"\\pred_"+file, torch.squeeze(pred, dim=0), sr)
