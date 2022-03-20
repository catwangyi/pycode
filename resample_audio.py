import os
import librosa
import soundfile

if __name__ == "__main__":
    for root, dirs, files in os.walk("E:\dataset\\voicebank"):
        if len(dirs) == 0:
            if "test" in root:
                for file in files:
                    signal, sr = librosa.load(os.path.join(root, file), sr=16000)
                    soundfile.write(os.path.join(root, file), signal, samplerate=sr)
