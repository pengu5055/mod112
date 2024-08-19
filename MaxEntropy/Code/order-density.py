"""
Plot filter behavior for different orders and evaluation densities.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from yw import YuleWalker
import cmasher as cmr
from time import time

# Use custom style
mpl.style.use('./ma-style.mplstyle')
colors = cmr.take_cmap_colors('cmr.tropical', 8, cmap_range=(0.0, 0.85))

# Load the signal
# data_path = "./SuppliedData/val3.dat"
# sig = np.genfromtxt(data_path)
data_path = "./MaxEntropy/Data/co2.dat"
with np.load("./MaxEntropy/Data/co2-detrended.npz") as data:
    sig = data["detrended"]
N_eval = [16, 32, 64, 128, 256, 512, 1024, 2048]

t_s = time()
fft_spectra = np.abs(np.fft.fft(sig))**2
fft_spectra = fft_spectra[:len(fft_spectra)//2]
fft_spectra = fft_spectra / np.max(fft_spectra)
t_fft = time() - t_s

# Interpolate or truncate data to fit the evaluation density
fft_s = {}
for N in N_eval:
    fft_s[N] = np.interp(np.linspace(0, np.pi, N, endpoint=False), np.linspace(0, np.pi, len(fft_spectra), endpoint=False), fft_spectra)


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

spectra = {}
times = []
for f, yw in filters.items():
    row = []
    times_row = []
    for N in N_eval:
        t_s = time()
        yw._freq(N=N)
        w, psd = yw.psd()
        psd = np.abs(psd)**2
        psd = psd / np.max(psd)
        row.append(psd)
        times_row.append(time() - t_s)
    row.append(w)
    times.append(times_row)
    spectra[f] = row

times = np.array(times) / t_fft

abs_diff = []
for f, yw in filters.items():
    row = []
    for i in range(len(N_eval)):
        row.append(np.abs(spectra[f][i] - fft_s[N_eval[i]]).mean())

    abs_diff.append(row)
abs_diff = np.array(abs_diff)

poly = {}
for f, yw in filters.items():
    poly[f] = np.roots(np.r_[1, -yw.R])
        
# Plot Abs. diff from FFT
fig, ax = plt.subplots(1, 3, figsize=(14, 4.5), layout="compressed")
y_ticks = [int(f) for f in filters.keys()]

cmap = cmr.get_sub_cmap('cmr.tropical', 0.0, 0.85)
norm = mpl.colors.LogNorm(vmin=abs_diff.min(), vmax=abs_diff.max())
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
ax[0].imshow(abs_diff, cmap=cmap, norm=norm, aspect='auto', origin='lower', zorder=2,
             extent=[0, len(filters), 0, len(N_eval)])
cbar = fig.colorbar(sm, ax=ax[0], orientation='vertical', pad=0.015
                    )
cbar.set_label("Mean Absolute Difference")
ax[0].set_title("Absolute Difference from FFT")
ax[0].set_ylabel("Evaluation Density")
ax[0].set_xlabel("Filter Order")
ax[0].set_xticks(np.arange(0, len(filters), 1) + 1/2)
ax[0].set_xticklabels(y_ticks)
ax[0].set_yticks(np.arange(0, len(N_eval), 1) + 1/2)
ax[0].set_yticklabels(N_eval)


ax[1].set_title("Computation Time vs. FFT")
norm = mpl.colors.LogNorm(vmin=times.min(), vmax=times.max())
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
ax[1].imshow(times, cmap=cmap, norm=norm, aspect='auto', origin='lower', zorder=2,
                extent=[0, len(filters), 0, len(N_eval)])
props = dict(boxstyle='round', facecolor='black', alpha=0.2)
textstr = f"FFT Evaluation Time: {t_fft:.2e} s"
ax[1].text(0.03, 0.95, textstr, transform=ax[1].transAxes, fontsize=12,
              verticalalignment='top', color="white", zorder=10, bbox=props)
cbar = fig.colorbar(sm, ax=ax[1], orientation='vertical', pad=0.015)
cbar.set_label("Relative Time to FFT")
ax[1].set_ylabel("Evaluation Density")
ax[1].set_xlabel("Filter Order")
ax[1].set_xticks(np.arange(0, len(filters), 1) + 1/2)
ax[1].set_xticklabels(y_ticks)
ax[1].set_yticks(np.arange(0, len(N_eval), 1) + 1/2)
ax[1].set_yticklabels(N_eval)

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

plt.suptitle(f"Filter Behavior for Different Orders and Evaluation Densities on {data_path.split('/')[-1]}")
plt.savefig(f"./MaxEntropy/Images/order-density-{data_path.split('/')[-1]}.pdf", dpi=500)
plt.show()

