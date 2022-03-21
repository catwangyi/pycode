from util.utils import *
import torch
import torchaudio
from model import AECNN_T
import soundfile


if __name__ == "__main__":
    state_dict = torch.load('best_model.pth', map_location='cpu')
    net = AECNN_T()
    hop_len = 256
    net.load_state_dict(state_dict)
    input_sig, sr = torchaudio.load('E:\dataset\\voicebank\\test\\noisy_testset_wav\p232_187.wav')
    needed_len = np.ceil(((input_sig.shape[-1])-2048) / hop_len) * hop_len + 2048
    noisy_sig = padsignal(input_sig=input_sig, needed_len=int(needed_len))
    noisy_frames = audio_to_frame(audio=noisy_sig,
                                  frame_size=2048,
                                  hop_len=hop_len,
                                  window=torch.ones(2048),
                                  device='cpu')
    pred = torch.empty_like(noisy_frames)
    for frame_idx in range(noisy_frames.shape[-1]):
        input = torch.reshape(noisy_frames[:, frame_idx], (1, 1, 2048))
        net_out = torch.squeeze(net(input), dim=0)
        pred[:, frame_idx] = net_out

    audio = frame_to_audio(pred,
                           frame_size=pred.shape[0],
                           frame_num=pred.shape[1],
                           hop_len=256,
                           window=torch.ones(2048),
                           device='cpu')
    audio = audio.permute(1, 0)
    pred_audio = torch.squeeze(audio).detach().numpy()
    soundfile.write('pred.wav', data=pred_audio, samplerate=sr)


