import torch
import torchaudio
import numpy
import librosa


if __name__ == "__main__":
    audio = torch.ones(1024)
    spec_func = torchaudio.transforms.Spectrogram(n_fft=512,
                                                  hop_length=256,
                                                  onesided=False,
                                                  power=None,
                                                  center=False,
                                                  return_complex=True
                                                  )
    spec = spec_func(audio)
    a = 1
