import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import noisereduce as nr
import scipy.signal as signal

# Load Audio File
def load_audio(file_path, sr=16000):
    y, sr = librosa.load(file_path, sr=sr)
    return y, sr

# Noise Reduction using Spectral Gating
def reduce_noise(y, sr):
    reduced_noise = nr.reduce_noise(y=y, sr=sr, stationary=True)
    return reduced_noise

# Bandpass Filter (500Hz - 2000Hz for lung sounds)
def bandpass_filter(y, sr, lowcut=500, highcut=2000, order=5):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_audio = signal.filtfilt(b, a, y)
    return filtered_audio
 
# Remove Silence
def remove_silence(y, sr, threshold=20):
    intervals = librosa.effects.split(y, top_db=threshold)
    non_silent_audio = np.concatenate([y[start:end] for start, end in intervals])
    return non_silent_audio

# Convert to Mel-Spectrogram
def extract_mel_spectrogram(y, sr, n_mels=128):
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

# Convert to MFCCs
def extract_mfcc(y, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc

# Plot Mel-Spectrogram
def plot_mel_spectrogram(mel_spec_db, sr):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Mel-Spectrogram")
    plt.show()

# Example Usage
file_path = "Audio Files\BP1_Asthma,I E W,P L L,70,M.wav" 
y, sr = load_audio(file_path)
y_denoised = reduce_noise(y, sr)
y_filtered = bandpass_filter(y_denoised, sr)
y_clean = remove_silence(y_filtered, sr)

mel_spectrogram = extract_mel_spectrogram(y_clean, sr)
mfcc_features = extract_mfcc(y_clean, sr)

plot_mel_spectrogram(mel_spectrogram, sr)
