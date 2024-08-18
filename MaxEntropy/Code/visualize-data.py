"""
Visualize the given data and try the YuleWalker class.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr
from yw import YuleWalker

# Use custom style
mpl.style.use("./ma-style.mplstyle")
colors = cmr.take_cmap_colors('cmr.tropical', 8, cmap_range=(0, 0.85))

# Load data
load_file = "./SuppliedData/val2.dat"
sig = np.genfromtxt(load_file)

# Plot the data
fig, ax = plt.subplots(1, 2, figsize=(12, 5), layout="compressed")

ax[0].plot(sig, color=colors[2])
ax[0].set_title(f"Signal {load_file.split('/')[-1]}")
ax[0].set_xlabel("Sample")
ax[0].set_ylabel("Amplitude")

# Estimate the AR model (w/ 2pi rad/sample sample frequency)
order = 10
yw = YuleWalker(sig, order)
w, psd = yw.psd()

# Get FFT of the signal
fft_sig = np.fft.fft(sig)
fft_sig = fft_sig[:len(fft_sig)//2]
psd_sig = np.abs(fft_sig)**2

# Plot the PSD
ax[1].plot(w, psd, color=colors[1], label=f"AR Model w/ {order}th Order Yule-Walker",
           lw=3, zorder=2)
ax[1].plot(np.linspace(0, np.pi, len(psd_sig), endpoint=False), psd_sig, color=colors[7],
           label="FFT of Signal", lw=2, zorder=3)
ax[1].set_title("Power Spectral Density")
ax[1].set_xlabel("Frequency [rad/sample]")
ax[1].set_ylabel("Power")
ax[1].legend()

plt.savefig(f"./MaxEntropy/Images/yw-demo-{load_file.split('/')[-1]}.pdf", dpi=500)
plt.show()
