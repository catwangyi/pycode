import torch
import torchaudio, os
from WaveUnet.model import WaveUnet

if __name__ == "__main__":
    IN_CHANNEL_LIST = [1, 24, 28, 72, 96, 120, 144, 168, 192, 216, 240, 264]
    OUT_CHANNEL_LIST = [24, 28, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288]
    # unet = WaveUnet.model.WaveUnet(IN_CHANNEL_LIST, OUT_CHANNEL_LIST)
    # torch.save(unet, "test.pth")
    net = WaveUnet(IN_CHANNEL_LIST, OUT_CHANNEL_LIST)
    net.load_state_dict(torch.load("best_model", map_location='cpu'))
    net.eval()
    with torch.no_grad():
        for root, dirs, files in os.walk("audio\\noisy"):
            for file in files:
                input_sig, sr = torchaudio.load(root+"\\"+file)
                input_sig = torch.unsqueeze(input_sig, dim=0)
                pred = net(input_sig)
                a = input_sig.detach().numpy()
                b = pred.detach().numpy()
                torchaudio.save("audio\pred"+"\\pred_"+file, torch.squeeze(pred, dim=0), sr)
