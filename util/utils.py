import torch
import numpy as np


def DFT_matrix(N):
    # i, j = np.meshgrid(np.arange(N), np.arange(N))
    # omega = np.exp( - 2 * np.pi * 1J / N )
    # W = np.power( omega, i * j ) / np.sqrt(N)
    w = np.empty((N, N), dtype=complex)
    for k in range(N):
        for n in range(N):
            w[k][n] = np.exp(-1j*(2*np.pi/N)*n*k)
    return w


def frame_to_audio(input_sig, shift=256, device='cuda'):
    '''
    :param time_domain_signal:shape(frame_size, frame_num)
    :return: OLA_out
    '''
    frame_size = input_sig.shape[0]
    frame_num = input_sig.shape[1]
    audio = torch.empty((frame_num-1)*shift+frame_size, 1, device=device)
    for i in range(0, frame_num):
        a = input_sig[:, i] * torch.hamming_window(frame_size, device=device)
        audio[i*shift:i*shift+frame_size] = torch.unsqueeze(a, dim=-1)
    return audio


def padsignal(input_sig, needed_len):
    sig = input_sig
    if input_sig.shape[-1] < needed_len:
        rest_len = needed_len - input_sig.shape[-1]
        sig = np.pad(array=input_sig, pad_width=(0, rest_len), mode="constant", constant_values=0)
    elif input_sig.shape[-1] > needed_len:
        start_idx = 0
        sig = input_sig[start_idx:start_idx + needed_len]
    return sig


def audio_to_frame(audio, frame_size=512, window=torch.hann_window(512), shift=256, device='cpu'):
    '''

    :param audio: [audio_len, 1]
    :param window:
    :return:
    '''
    audio = torch.squeeze(audio)
    window = window.to(device)
    assert (len(audio)-frame_size) / shift % 1 == 0
    frame_num = int((len(audio)-frame_size) / shift) + 1
    frames = torch.empty((frame_size, frame_num), device=device)
    for i in range(frame_num):
        frames[:, i] = audio[i*shift:i*shift+frame_size] * window
    return frames