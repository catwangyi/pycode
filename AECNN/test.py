from util.utils import *
import torch
import torchaudio
from model import AECNN_T
import soundfile
'''
test reconstruct audio from frames
'''


if __name__ == "__main__":
    sig, sr = torchaudio.load('E:\dataset\\voicebank\\test\\clean_testset_wav\p232_001.wav')
    hop_len = 256
    needed_len = np.ceil(((sig.shape[-1])-2048) / hop_len) * hop_len + 2048
    clean_sig = padsignal(input_sig=sig, needed_len=int(needed_len))

    # noisy_frames is the input of network, so the output should be like clean_frames
    clean_frames = audio_to_frame(audio=clean_sig,
                                  frame_size=2048,
                                  window=torch.ones(2048),
                                  hop_len=hop_len,
                                  device='cpu')
    audio = frame_to_audio(clean_frames,
                           frame_size=clean_frames.shape[0],
                           window=torch.ones(2048),
                           frame_num=clean_frames.shape[1],
                           hop_len=hop_len,
                           device='cpu')
    audio = audio.permute(1, 0)
    pred_audio = torch.squeeze(audio).detach().numpy()
    soundfile.write('pred.wav', data=pred_audio, samplerate=sr)


