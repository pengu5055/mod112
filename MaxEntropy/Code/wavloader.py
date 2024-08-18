"""
Contains the functions for the analysis of the data.
"""
import numpy as np
from scipy.io import wavfile

def load_wav(filename):
    fs_rate, signal = wavfile.read(filename)
    # Average stereo channels
    if len(signal.shape) == 2:
        signal = signal.sum(axis=1)/2
    # Read number of samples, duration and sampling frequency
    n = signal.shape[0]
    secs = n / fs_rate
    T = 1/fs_rate
    t = np.arange(0, secs, T)
    data = {
        'fs_rate': fs_rate,
        'signal': signal,
        'n': n,
        'secs': secs,
        'T': T,
        't': t
    }
    
    return data