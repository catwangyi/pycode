import librosa
import os

import soundfile

if __name__ == "__main__":
    noisy_list = []
    for root, dirs, files in os.walk("E:\dataset\DEMAND"):
        if len(dirs) == 0:
            for file in files:
                noisy_sig, sr = librosa.load(os.path.join(root, file), sr=16000)
                noisy_list.append(noisy_sig)
            noisy_all_channel = 0
            for noisy in noisy_list:
                noisy_all_channel += noisy
            soundfile.write(root + "\\" + "merged.wav", noisy_all_channel / 16, sr)
