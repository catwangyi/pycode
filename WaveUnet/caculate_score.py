import numpy as np

from pystoi import stoi

import librosa
import os
from pesq import pesq


def caculate_stoi(clean_file_list, denoised_file_list):
    score = 0
    for clean_path, enhanced_path in zip(clean_file_list, denoised_file_list):
        clean, sr = librosa.load(clean_path, sr=None)
        enhanced, _ = librosa.load(enhanced_path, sr=sr)

        x = np.floor(np.log2(sr * 5)) + 1
        needed_len = int(2 ** x)
        if clean.shape[-1] <= needed_len:
            rest_len = needed_len - clean.shape[-1]
            clean = np.pad(array=clean, pad_width=(0, rest_len), mode="constant", constant_values=0)
            enhanced = np.pad(array=enhanced, pad_width=(0, rest_len), mode="constant", constant_values=0)
        else:
            start_idx = int((clean.shape[-1] - needed_len) / 2)
            clean = clean[:, start_idx:start_idx + needed_len]
            enhanced = enhanced[:, start_idx:start_idx + needed_len]

        score += stoi(clean, enhanced, sr)
    print("stoi:", score / len(clean_file_list))


def caculate_pesq(clean_file_list, denoised_file_list):
    score = 0
    for clean_path, enhanced_path in zip(clean_file_list, denoised_file_list):
        clean, sr = librosa.load(clean_path, sr=16000)
        enhanced, _ = librosa.load(enhanced_path, sr=sr)
        x = np.floor(np.log2(sr * 5)) + 1
        needed_len = int(2 ** x)
        if clean.shape[-1] <= needed_len:
            rest_len = needed_len - clean.shape[-1]
            clean = np.pad(array=clean, pad_width=(0, rest_len), mode="constant", constant_values=0)
            enhanced = np.pad(array=enhanced, pad_width=(0, rest_len), mode="constant", constant_values=0)
        else:
            start_idx = int((clean.shape[-1] - needed_len) / 2)
            clean = clean[:, start_idx:start_idx + needed_len]
            enhanced = enhanced[:, start_idx:start_idx + needed_len]

        score += pesq(sr, clean, enhanced)
        # print(score)
    print("pesq:", score / len(clean_file_list))


if __name__ == "__main__":
    clean_file_list = []
    denoised_file_list = []
    for root, dirs, files in os.walk("D:\\voicebank\\test\clean_testset_wav"):
        for file in files:
            clean_file_list.append(os.path.join(root, file))

    for root, dirs, files in os.walk("D:\\voicebank\\test\\noisy_testset_wav"):
        for file in files:
            denoised_file_list.append(os.path.join(root, file))
    print("before enhancement:")
    caculate_stoi(clean_file_list, denoised_file_list)
    caculate_pesq(clean_file_list, denoised_file_list)


    denoised_file_list = []
    for root, dirs, files in os.walk("D:\\voicebank\pred"):
        for file in files:
            denoised_file_list.append(os.path.join(root, file))
    print("after enhancement:")
    caculate_stoi(clean_file_list, denoised_file_list)
    caculate_pesq(clean_file_list, denoised_file_list)
