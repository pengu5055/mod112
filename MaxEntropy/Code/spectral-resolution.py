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
import scipy.signal as signal
import pickle

# Use custom style
mpl.style.use('./ma-style.mplstyle')
colors = cmr.take_cmap_colors('cmr.tropical', 8, cmap_range=(0.0, 0.85))

# Create the signal
signals = []
peak_distances = np.linspace(0.1, 1, 100)  # In rad/sample
x = np.linspace(0, np.pi, 4096)
for d in peak_distances:
    sig = norm.pdf(x, loc=0.5, scale=0.05) + norm.pdf(x, loc=0.5+d, scale=0.05)
    signals.append(sig)

t_s = time()
fft_spectra = np.abs(np.fft.fft(sig))**2
fft_spectra = fft_spectra[:len(fft_spectra)//2]
fft_spectra = fft_spectra / np.max(fft_spectra)
t_fft = time() - t_s

N_eval = [16, 32, 64, 128, 256, 512, 1024, 2048]
orders = [2, 4, 8, 16, 32, 64, 128, 256, 512]
if True:
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
                peaks_row.append(signal.find_peaks(psd, height=30)[0])
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



print("yes boss")


        
# Plot Abs. diff from FFT
fig, ax = plt.subplots(1, 3, figsize=(14, 4.5), layout="compressed")
y_ticks = [int(f) for f in filters.keys()]

cmap = cmr.get_sub_cmap('cmr.tropical', 0.0, 0.85)
norm = mpl.colors.LogNorm(vmin=abs_diff.min(), vmax=abs_diff.max())
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
ax[0].imshow(abs_diff, cmap=cmap, norm=norm, aspect='auto', origin='lower', zorder=2,
             extent=[0, len(N_eval), 0, len(filters)])
cbar = fig.colorbar(sm, ax=ax[0], orientation='vertical', pad=0.015
                    )
cbar.set_label("Mean Absolute Difference")
ax[0].set_title("Absolute Difference from FFT")
ax[0].set_xlabel("Evaluation Density")
ax[0].set_ylabel("Filter Order")
ax[0].set_yticks(np.arange(0, len(filters), 1) + 1/2)
ax[0].set_yticklabels(y_ticks)
ax[0].set_xticks(np.arange(0, len(N_eval), 1) + 1/2)
ax[0].set_xticklabels(N_eval)


ax[1].set_title("Computation Time vs. FFT")
norm = mpl.colors.LogNorm(vmin=times.min(), vmax=times.max())
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
ax[1].imshow(times, cmap=cmap, norm=norm, aspect='auto', origin='lower', zorder=2,
                extent=[0, len(N_eval), 0, len(filters)])
props = dict(boxstyle='round', facecolor='black', alpha=0.2)
textstr = f"FFT Evaluation Time: {t_fft:.2e} s"
ax[1].text(0.03, 0.95, textstr, transform=ax[1].transAxes, fontsize=12,
              verticalalignment='top', color="white", zorder=10, bbox=props)
cbar = fig.colorbar(sm, ax=ax[1], orientation='vertical', pad=0.015)
cbar.set_label("Relative Time to FFT")
ax[1].set_xlabel("Evaluation Density")
ax[1].set_ylabel("Filter Order")
ax[1].set_yticks(np.arange(0, len(filters), 1) + 1/2)
ax[1].set_yticklabels(y_ticks)
ax[1].set_xticks(np.arange(0, len(N_eval), 1) + 1/2)
ax[1].set_xticklabels(N_eval)

ax[2].set_title("Char. Polynomial Roots")
ax[2].set_xlabel("Real")
ax[2].set_ylabel("Imaginary")
for i, a in enumerate(poly.items()):
    f, p = a
    ax[2].scatter(np.real(p), np.imag(p), label=f"Order {f}", color=colors[i], alpha=0.6)
ax[2].legend(ncols=3, loc="upper left", fontsize=8)
ax[2].plot(np.cos(np.linspace(0, 2*np.pi, 100)), np.sin(np.linspace(0, 2*np.pi, 100)), color="k", lw=3, ls="--", zorder=2)
ax[2].set_xlim(-1.5, 1.5)
ax[2].set_ylim(-1.5, 1.5)


# Make frame between the cells of the heatmap
for a in ax.flatten()[:-1]:
    N_box = np.arange(0, len(N_eval) + 1, 1)
    filter_box = np.arange(0, len(filters) + 1, 1)
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

plt.suptitle(f"Filter Behavior for")
plt.savefig(f"./MaxEntropy/Images/s-res.pdf", dpi=500)
plt.show()

