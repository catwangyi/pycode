from pystoi import stoi
import librosa
import os
from pesq import pesq


def caculate_stoi(clean_file_list, denoised_file_list):
    for clean_path, enhanced_path in zip(clean_file_list, denoised_file_list):
        clean, sr = librosa.load(clean_path, sr=None)
        enhanced, _ = librosa.load(enhanced_path, sr=None)
        score = stoi(clean, enhanced, sr)
        print(score)


def caculate_pesq(clean_file_list, denoised_file_list, mode='wb'):
    for clean_path, enhanced_path in zip(clean_file_list, denoised_file_list):
        clean, sr = librosa.load(clean_path, sr=16000)
        enhanced, _ = librosa.load(enhanced_path, sr=None)
        score = pesq(sr, clean, enhanced, mode=mode)
        print(score)


if __name__ == "__main__":
    clean_file_list = []
    denoised_file_list = []
    for root, dirs, files in os.walk("audio\clean"):
        for file in files:
            clean_file_list.append(os.path.join(root, file))

    for root, dirs, files in os.walk("audio\pred"):
        for file in files:
            denoised_file_list.append(os.path.join(root, file))

    caculate_pesq(clean_file_list, denoised_file_list, mode='nb')
