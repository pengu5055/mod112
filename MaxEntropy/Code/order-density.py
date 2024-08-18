"""
Plot filter behavior for different orders and evaluation densities.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from yw import YuleWalker
import cmasher as cmr

# Use custom style
mpl.style.use('./ma-style.mplstyle')
colors = cmr.take_cmap_colors('cmr.tropical', 8, cmap_range=(0.0, 0.85))

# Load the signal
data_path = "./SuppliedData/val2.dat"
sig = np.genfromtxt(data_path)
N_eval = [16, 32, 64, 128, 256, 512, 1024, 2048]

fft_spectra = np.abs(np.fft.fft(sig))**2
fft_spectra = fft_spectra[:len(fft_spectra)//2]
fft_spectra = fft_spectra / np.max(fft_spectra)

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
for f, yw in filters.items():
    row = []
    for N in N_eval:
        yw._freq(N=N)
        w, psd = yw.psd()
        psd = np.abs(psd)**2
        psd = psd / np.max(psd)
        row.append(psd)
    row.append(w)
    spectra[f] = row

abs_diff = []
for f, yw in filters.items():
    row = []
    for i in range(len(N_eval)):
        row.append(np.abs(spectra[f][i] - fft_s[N_eval[i]]).mean())

    abs_diff.append(row)
abs_diff = np.array(abs_diff)
print(abs_diff.shape)
        
# Plot Abs. diff from FFT
fig, ax = plt.subplots(1, 3, figsize=(14, 5), layout="compressed")

cmap = cmr.get_sub_cmap('cmr.tropical', 0.0, 0.85)
norm = mpl.colors.Normalize(vmin=abs_diff.min(), vmax=abs_diff.max())
ax[0].imshow(abs_diff, cmap=cmap, norm=norm, aspect='auto', origin='lower',
             extent=[N_eval[0], N_eval[-1], 0, len(filters)])
ax[0].set_title("Absolute Difference from FFT")
ax[0].set_xlabel("Evaluation Density")
ax[0].set_ylabel("Filter Order")
# ax[0].set_yticks()


plt.show()

