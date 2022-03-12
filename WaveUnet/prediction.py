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
                pred = net(torch.unsqueeze(input_sig, dim=0))
                pred = torch.squeeze(pred, dim=0)
                torchaudio.save("audio\pred"+"\\pred_"+file, pred, sr)
