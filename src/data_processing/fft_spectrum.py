import os
import numpy as np

fs_ppg = 64
fs_acc = 32

def fft_and_normalize(segment, target_len=1025, fs = None):
    min_n_fft = (target_len - 1) * fs / 4.0

    n_fft = 1
    while n_fft < min_n_fft:
        n_fft *= 2

    padded = np.zeros(int(n_fft))
    padded[:len(segment)] = segment.flatten()

    spectrum = np.abs(np.fft.fft(padded))

    spectrum = spectrum[:target_len]

    mean = np.mean(spectrum)
    std = np.std(spectrum)

    if std == 0:
        return np.zeros_like(spectrum)
    return (spectrum - mean) / std

