import librosa


if __name__ == "__main__":
    signal, sr = librosa.load("../audio/1.wav", sr=None)
    a = 1
