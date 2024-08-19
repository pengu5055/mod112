"""
Plot spectrum resolution for different filter orders and peak distances.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from yw import YuleWalker
import cmasher as cmr
from time import time
from scipy.stats import norm
import scipy.fft as fft
import scipy.signal as signal
import pickle

# Use custom style
mpl.style.use('./ma-style.mplstyle')
colors = cmr.take_cmap_colors('cmr.tropical', 8, cmap_range=(0.0, 0.85))

# Create the signal
signals = []
peak_distances = np.linspace(0.001, 0.01, 8)  # In rad/sample
x = np.linspace(0, 1, 4096)
for d in peak_distances:
    # sig = norm.pdf(x, loc=0.5, scale=0.05) + norm.pdf(x, loc=0.5+d, scale=0.05)
    sig = np.sin(5000 * np.pi * x) + np.sin(5000*(1+d) * np.pi * x)
    signals.append(sig)

d = fft.fft(signals[0])[:2048]
d = np.abs(d)**2
d = d / np.max(d)
data_len = len(x)
SAMPLE_RATE = 2*np.pi
data_time = 1/SAMPLE_RATE * data_len
freq = np.linspace(0.0, 1.0/(2.0*(data_time/data_len)), data_len//2)

if False:
    cm = cmr.take_cmap_colors("cmr.tropical", len(peak_distances), cmap_range=(0.0, 0.85))
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for i, sig in enumerate(signals):
        d = fft.fft(sig)[:2048]
        d = np.abs(d)**2
        d = d / np.max(d)
        ax.plot(freq, d, label=f"Rel. Peak Distance: {peak_distances[i]:.3f}", color=cm[i])
    
    plt.suptitle("Spectra of Test Signals with Different Peak Distances")
    ax.set_xlabel("Frequency [rad/sample]")
    ax.set_ylabel("Power Spectrum Density")
    ax.legend()
    ax.set_xlim(2.3, 2.5)
    plt.savefig(f"./MaxEntropy/Images/test-spectra.pdf", dpi=500)
    plt.show()

t_s = time()
fft_spectra = np.abs(np.fft.fft(sig))**2
fft_spectra = fft_spectra[:len(fft_spectra)//2]
fft_spectra = fft_spectra / np.max(fft_spectra)
t_fft = time() - t_s
fft_peaks = signal.find_peaks(fft_spectra, height=0.05)[0]
fft_fwhm = signal.peak_widths(fft_spectra, fft_peaks, rel_height=0.5)[0]
fft_res = fft_peaks[1] - fft_peaks[0]

N_eval = [16, 32, 64, 128, 256, 512, 1024, 2048]
orders = [2, 4, 8, 16, 32, 64, 128, 256, 512]
if False:
    spectra = {}
    yw = {}
    times = []
    for o in orders:
        times_row1 = []
        for N in N_eval:
            times_row2 = []
            yw_row = []
            for sig in signals:
                t_s = time()
                y = YuleWalker(sig, order=16)
                y._freq(N=N)
                w, psd = y.psd()
                psd = np.abs(psd)**2
                psd = psd / np.max(psd)
                yw_row.append(psd)
                times_row2.append(time() - t_s)
            yw[N] = yw_row
            times_row1.append(times_row2)
        spectra[o] = yw
    
    with open('./MaxEntropy/Data/spectra.pickle', 'wb') as handle:
        pickle.dump(spectra, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./MaxEntropy/Data/times.pickle', 'wb') as handle:
        pickle.dump(times, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved data to ./MaxEntropy/Data/spectra.pickle")
    print(f"Saved data to ./MaxEntropy/Data/times.pickle")

    # Find Peaks and Their FWHM
    peaks = {}
    fwhm = {}
    for o, yw in spectra.items():
        peaks[o] = []
        fwhm[o] = []
        for N, sigs in yw.items():
            peaks_row = []
            fwhm_row = []
            for psd in sigs:
                peaks_row.append(signal.find_peaks(psd, height=0.05)[0])
                fwhm_row.append(signal.peak_widths(psd, peaks_row[-1], rel_height=0.5)[0])
            peaks[o].append(peaks_row)
            fwhm[o].append(fwhm_row)
    
    with open('./MaxEntropy/Data/peaks.pickle', 'wb') as handle:
        pickle.dump(peaks, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./MaxEntropy/Data/fwhm.pickle', 'wb') as handle:
        pickle.dump(fwhm, handle, protocol=pickle.HIGHEST_PROTOCOL)

    quit()
else:
    with open('./MaxEntropy/Data/spectra.pickle', 'rb') as handle:
        spectra = pickle.load(handle)
    with open('./MaxEntropy/Data/times.pickle', 'rb') as handle:
        times = pickle.load(handle)
    with open('./MaxEntropy/Data/peaks.pickle', 'rb') as handle:
        peaks = pickle.load(handle)
    with open('./MaxEntropy/Data/fwhm.pickle', 'rb') as handle:
        fwhm = pickle.load(handle)

R_power = []
for o, f in peaks.items():
    R_power_row2 = []
    for sigs in f:
        R_power_row = []
        for p in sigs:
            if len(p) <= 1:
                R_power_row.append(0)
            else:
                R_power_row.append((p[1] - p[0])/fft_res)
        R_power_row2.append(R_power_row)
    R_power.append(R_power_row2)

if False:
    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    for j, s in enumerate([2, 16, 32]):
        for i in range(8):
            ax[j].plot(freq, spectra[s][2048][i], color=colors[i], lw=2,
                       label=f"Distance: {peak_distances[i]:.3f}")
            ax[j].set_title(f"Order: {s}")
            ax[j].set_xlabel("Frequency [rad/sample]")
            ax[j].set_ylabel("Power Spectrum Density")
            ax[j].set_xlim(2.3, 2.5)
            ax[j].legend(ncols=1, loc="upper left", fontsize=8)

    plt.suptitle("AR Spectra w/ Different Peak Distances")
    plt.savefig(f"./MaxEntropy/Images/ar-spectra.pdf", dpi=500)
    plt.tight_layout()
    plt.show()

        
# Plot Abs. diff from FFT
fig, ax = plt.subplots(3, 3, figsize=(10, 9))
y_ticks = peak_distances

cmap = cmr.get_sub_cmap('cmr.tropical', 0.0, 0.85)
norm = mpl.colors.Normalize(vmin=0, vmax=1)
for i, f in enumerate(R_power):
    ax.flatten()[i].imshow(f, cmap=cmap, aspect="auto", origin="lower",
                           extent=[0, len(N_eval), 0, len(peak_distances)], zorder=2)
    ax.flatten()[i].set_xticks(np.arange(len(N_eval)) + 1/2)
    ax.flatten()[i].set_xticklabels(N_eval, rotation=35)
    ax.flatten()[i].set_yticks(np.arange(len(peak_distances)) + 1/2)
    ax.flatten()[i].set_yticklabels([f"{p:.3f}" for p in peak_distances])
    ax.flatten()[i].set_xlabel("Evaluation Density")
    ax.flatten()[i].set_ylabel("Peak Distance")
    ax.flatten()[i].set_title(f"Order: {orders[i]}")
    ax.flatten()[i].grid(False)

cbar_ax = fig.add_axes([0.875, 0.08, 0.025, 0.8])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, cax=cbar_ax, aspect=20)
cbar.set_label("Relative Resolution Power vs. FFT", labelpad=10)

# Make frame between the cells of the heatmap
for a in ax.flatten():
    N_box = np.arange(0, len(N_eval) + 1, 1)
    filter_box = np.arange(0, len(peak_distances) + 1, 1)
    for i in range(len(N_box)):
        for j in range(len(filter_box)):
            x1 = np.array([N_box[i], N_box[i]])
            y1 = np.array([filter_box[j], filter_box[j-1]])
            x2 = np.array([N_box[i-1], N_box[i]])
            y2 = np.array([filter_box[j], filter_box[j]])
            a.plot(x1, y1, color="k", lw=0.5, zorder=6)
            a.plot(x2, y2, color="k", lw=0.5, zorder=6)
            a.plot([N_box[0], N_box[0]], [filter_box[0], filter_box[-1]], color="k", lw=4.5, zorder=6)
            a.plot([N_box[-1], N_box[-1]], [filter_box[0], filter_box[-1]], color="k", lw=4.5, zorder=6)
            a.plot([N_box[0], N_box[-1]], [filter_box[0], filter_box[0]], color="k", lw=3.5, zorder=6)
            a.plot([N_box[0], N_box[-1]], [filter_box[-1], filter_box[-1]], color="k", lw=3.5, zorder=6)

plt.suptitle(f"Resolution Power of AR vs. FFT")
plt.savefig(f"./MaxEntropy/Images/s-res.pdf", dpi=500)
plt.subplots_adjust(right=0.85, top=0.9, bottom=0.07, left=0.08, hspace=0.6, wspace=0.5)
plt.show()

