import numpy as np
import librosa
import soundfile
import os
import random


def add_noisy(file_path, noisy_path, dst_snr):
    clean, sr = librosa.load(file_path, sr=16000)
    noisy, _ = librosa.load(noisy_path, sr=sr)
    offset = random.randint(0, len(noisy) - len(clean))
    noisy = noisy[offset:len(clean)+offset]

    clean_power = np.sum(np.square(clean))
    noisy_power = np.sum(np.square(noisy))
    snr = 10 * np.log10(clean_power/noisy_power)
    # print("origin:", snr)
    # print("needed:", dst_snr)
    scale_factor = np.sqrt(10 ** ((snr-dst_snr) / 10))
    noisy = noisy * scale_factor

    # noisy_power = np.sum(np.square(noisy))
    # snr = 10 * np.log10(clean_power / noisy_power)
    # print("after:", snr)
    return clean + noisy


if __name__ =="__main__":
    noisy_file_path_list = []
    for root, dirs, files in os.walk("E:\dataset\DEMAND"):
        if len(dirs) == 0:
            noisy_file_path_list.append(os.path.join(root, "merged.wav"))
    snr_list = [2.5, 0, -2.5, -5]
    snr_list_idx = 0
    noisy_list_idx = 0
    for root, dirs, files in os.walk("D:\\voicebank\\train\clean_trainset_wav"):
        if len(dirs) == 0:
            for file in files:
                if snr_list_idx == len(snr_list):
                    snr_list_idx = 0
                if noisy_list_idx == len(noisy_file_path_list):
                    noisy_list_idx = 0
                noisy_data = add_noisy(os.path.join(root, file), noisy_file_path_list[noisy_list_idx], snr_list[snr_list_idx])
                snr_list_idx += 1
                noisy_list_idx += 1
                soundfile.write(os.path.join("D:\\voicebank\\train\\noisy_trainset_wav", file),
                                noisy_data,
                                samplerate=16000)
    snr_list_idx = 0
    noisy_list_idx = 0
    for root, dirs, files in os.walk("D:\\voicebank\\test\clean_testset_wav"):
        if len(dirs) == 0:
            for file in files:
                if snr_list_idx == len(snr_list):
                    snr_list_idx = 0
                if noisy_list_idx == len(noisy_file_path_list):
                    noisy_list_idx = 0
                noisy_data = add_noisy(os.path.join(root, file), noisy_file_path_list[noisy_list_idx], snr_list[snr_list_idx])
                snr_list_idx += 1
                noisy_list_idx += 1
                soundfile.write(os.path.join("D:\\voicebank\\test\\noisy_testset_wav", file),
                                noisy_data,
                                samplerate=16000)
