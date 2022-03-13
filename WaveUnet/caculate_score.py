import numpy as np

from pystoi import stoi

import librosa
import os
from pesq import pesq


def caculate_stoi(clean_file_list, denoised_file_list):
    score = 0
    for clean_path, enhanced_path in zip(clean_file_list, denoised_file_list):
        clean, sr = librosa.load(clean_path, sr=None, dtype='int16')
        enhanced, _ = librosa.load(enhanced_path, sr=sr, dtype='int16')
        score += stoi(clean, enhanced, sr)
    print(score / len(clean_file_list))


def caculate_pesq(clean_file_list, denoised_file_list):
    score = 0
    for clean_path, enhanced_path in zip(clean_file_list, denoised_file_list):
        clean, sr = librosa.load(clean_path, sr=16000)
        enhanced, _ = librosa.load(enhanced_path, sr=sr)
        score += pesq(sr, clean, enhanced)
        # print(score)
    print(score / len(clean_file_list))


if __name__ == "__main__":
    clean_file_list = []
    denoised_file_list = []
    for root, dirs, files in os.walk("D:\\voicebank\\test\clean_testset_wav"):
        for file in files:
            clean_file_list.append(os.path.join(root, file))

    for root, dirs, files in os.walk("D:\\voicebank\pred"):
        for file in files:
            denoised_file_list.append(os.path.join(root, file))

    caculate_pesq(clean_file_list, denoised_file_list)
