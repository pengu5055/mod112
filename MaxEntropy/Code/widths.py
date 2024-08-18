"""
Calculate and fit the peak widths of the spectral lines using when
using different windows.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal as signal
import scipy.fft as fft
from yw import YuleWalker
import cmasher as cmr

# Use custom style
mpl.style.use('./ma-style.mplstyle')
colors = cmr.take_cmap_colors('cmr.tropical', 8, cmap_range=(0.0, 0.85))


# Load the signal
data_path = "./SuppliedData/val3.dat"
sig = np.genfromtxt(data_path)

filters = {
    "2": YuleWalker(sig, 2),
    "4": YuleWalker(sig, 4),
    "8": YuleWalker(sig, 8),
    "16": YuleWalker(sig, 16),
    "32": YuleWalker(sig, 32),
    "64": YuleWalker(sig, 64),
    "128": YuleWalker(sig, 128),
    "256": YuleWalker(sig, 256),
}

spectra = {
    "FFT": np.abs(fft.fft(sig)[:len(sig)//2])**2,
}
peaks = {
    "FFT": signal.find_peaks(spectra["FFT"])[0],
}
peak_widths = {
    "FFT": signal.peak_widths(spectra["FFT"], peaks["FFT"], rel_height=0.1)[0] * 2 * np.pi / len(sig),
}

for f, yw in filters.items():
    _, spectra[f] = yw.psd()
    spectra[f] = np.abs(spectra[f])**2
    peaks[f], _ = signal.find_peaks(spectra[f], height=20)
    peak_widths[f] = signal.peak_widths(spectra[f], peaks[f], rel_height=0.2)
    peak_widths[f] = peak_widths[f][0] * 2 * np.pi / len(sig)

raw_peaks = peaks["FFT"].copy()
raw_peak_widths = peak_widths["FFT"].copy()

for i, f in peaks.items():
    try:
        peaks[i] = peaks[i] / raw_peaks
        peak_widths[i] = peak_widths[i] / raw_peak_widths
    except ValueError:
        pass

cm = cmr.take_cmap_colors('cmr.tropical', len(spectra), cmap_range=(0.0, 0.85))
fig, ax = plt.subplots(1, 2, figsize=(12, 5), layout='compressed')
x_axis = np.arange(len(spectra))

ax[0].set_title("Relative Inverse Peak Heights")
ax[0].bar(x_axis, [np.mean(peaks[i])**-1 for i in spectra], color=cm, alpha=1,
          zorder=10, edgecolor='black', linewidth=0.5)
ax[0].fill_between([-0.4, 0.4], 0, 1, color=cm[0], edgecolor="black", linewidth=0.5, hatch="\\\\", zorder=10)
ax[0].set_xticks(x_axis)
ax[0].set_xticklabels(spectra.keys(), rotation=30)
ax[0].set_yscale("log")
ax[0].set_ylabel("Inverse Peak Heights")


ax[1].set_title("Relative Peak Widths")
ax[1].bar(x_axis, [np.mean(peak_widths[i]) for i in spectra], color=cm, alpha=1,
          zorder=10, edgecolor='black', linewidth=0.5)
ax[1].set_xticks(x_axis)
ax[1].fill_between([-0.4, 0.4], 0, 1, color=cm[0], edgecolor="black", linewidth=0.5, hatch="\\\\", zorder=10)
ax[1].set_xticklabels(spectra.keys(), rotation=30)
ax[1].set_yscale("log")
ax[1].set_ylabel("Peak Widths")

plt.suptitle(f"Relative Peak Heights and Widths for AR Orders on {data_path.split('/')[-1]}")
plt.savefig(f"./MaxEntropy/Images/peak-widths-{data_path.split('/')[-1]}.pdf", dpi=500)
plt.show()
