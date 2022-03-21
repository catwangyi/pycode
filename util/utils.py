import torch
import numpy as np
import torch.nn.functional as F

def DFT_matrix(N):
    # i, j = np.meshgrid(np.arange(N), np.arange(N))
    # omega = np.exp( - 2 * np.pi * 1J / N )
    # W = np.power( omega, i * j ) / np.sqrt(N)
    w = np.empty((N, N), dtype=complex)
    for k in range(N):
        for n in range(N):
            w[k][n] = np.exp(-1j*(2*np.pi/N)*n*k)
    return w


def frame_to_audio(input_sig, frame_size, frame_num, window, hop_len=256, device='cuda'):
    '''
    :param time_domain_signal:shape(frame_size, frame_num)
    :return: OLA_out
    '''
    window = window.to(device)
    audio = torch.empty((frame_num-1) * hop_len + frame_size, 1, device=device)
    for i in range(0, frame_num):
        a = input_sig[:, i] * window
        audio[i*hop_len:i*hop_len+frame_size] = torch.unsqueeze(a, dim=-1)
    return audio


def padsignal(input_sig, needed_len):
    sig = input_sig
    # print('input_sig', input_sig.shape)
    # print('needed_len:', needed_len)
    if input_sig.shape[-1] < needed_len:
        rest_len = needed_len - input_sig.shape[-1]
        sig = F.pad(input_sig, pad=(0, rest_len), mode="constant", value=0)
    elif input_sig.shape[-1] > needed_len:
        start_idx = 0
        sig = input_sig[start_idx:start_idx + needed_len]
    # print('sig', sig.shape)
    return sig


def audio_to_frame(audio, frame_size=512, window=torch.hann_window(512), hop_len=256, device='cpu'):
    '''

    :param audio: [audio_len, 1]
    :param window:
    :return:
    '''
    audio = torch.squeeze(audio).to(device)
    window = window.to(device)
    # ensure the audio can be divided into integer num frames
    assert (len(audio)-frame_size) / hop_len % 1 == 0
    frame_num = int((len(audio)-frame_size) / hop_len) + 1
    frames = torch.empty((frame_size, frame_num), device=device)
    for i in range(frame_num):
        frames[:, i] = audio[i*hop_len:i*hop_len+frame_size] * window
    return frames