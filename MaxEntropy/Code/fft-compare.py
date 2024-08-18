"""
Compare the FFT of the signal with the AR model's PSD.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr
import scipy.fft as fft
from yw import YuleWalker

# Use custom style
mpl.style.use("./ma-style.mplstyle")
colors = cmr.take_cmap_colors('cmr.tropical', 8, cmap_range=(0, 0.85))

# Load data
data1 = np.genfromtxt("./SuppliedData/val2.dat")
data2 = np.genfromtxt("./SuppliedData/val3.dat")
with np.load("./MaxEntropy/Data/co2-detrended.npz") as data:
    data3_y = data["years"]
    data3 = data["detrended"]

# Calculate the FFT of the signals
fft_data1 = fft.fft(data1)
fft_data1 = np.abs(fft_data1[:len(fft_data1)//2])**2
fft_data2 = fft.fft(data2)
fft_data2 = np.abs(fft_data2[:len(fft_data2)//2])**2
fft_data3 = fft.fft(data3)
fft_data3 = np.abs(fft_data3[:len(fft_data3)//2])**2

fft_data1 = fft_data1 / np.max(fft_data1)
fft_data2 = fft_data2 / np.max(fft_data2)
fft_data3 = fft_data3 / np.max(fft_data3)

# Calculate AR model PSD
order1 = 10
yw1 = YuleWalker(data1, order1)
yw1._freq(N=256)
w1, psd1 = yw1.psd()

order2 = 10
yw2 = YuleWalker(data2, order2)
yw2._freq(N=256)
w2, psd2 = yw2.psd()

order3 = 10
yw3 = YuleWalker(data3, order3)
yw3._freq(N=305)
w3, psd3 = yw3.psd()

psd1 = psd1 / np.max(psd1)
psd2 = psd2 / np.max(psd2)
psd3 = psd3 / np.max(psd3)

# fft_data1 = fft_data1 * np.max(psd1)
# fft_data2 = fft_data2 * np.max(psd2)
# fft_data3 = fft_data3 * np.max(psd3)

# Plot the FFT, AR model PSD, the abs. diff. and the poles of the AR model
fig, ax = plt.subplots(3, 3, figsize=(12, 8), layout="compressed", sharex="col")

ax[0, 0].plot(w1, psd1, color=colors[1], label=f"AR Model w/ {order1}th\nOrder Yule-Walker",
              lw=3, zorder=2)
ax[0, 0].plot(np.linspace(0, np.pi, len(fft_data1), endpoint=False), fft_data1, ls="--",
                color=colors[7], label="FFT of Signal", lw=2, zorder=3)
ax[0, 0].legend(fontsize=8)

ax[0, 1].plot(w2, psd2, color=colors[2], label=f"AR Model w/ {order2}th\nOrder Yule-Walker",
                lw=3, zorder=2)
ax[0, 1].plot(np.linspace(0, np.pi, len(fft_data2), endpoint=False), fft_data2, ls="--",
                color=colors[7], label="FFT of Signal", lw=2, zorder=3)
ax[0, 1].legend(fontsize=8)

ax[0, 2].plot(w3, psd3, color=colors[4], label=f"AR Model w/ {order3}th\nOrder Yule-Walker",
                lw=3, zorder=2)
ax[0, 2].plot(np.linspace(0, np.pi, len(fft_data3), endpoint=False), fft_data3, ls="--",
                color=colors[7], label="FFT of Signal", lw=2, zorder=3)
ax[0, 2].legend(fontsize=8)

# Interpolate FFT to 512 points

# Plot the absolute difference
diff1 = np.abs(psd1 - fft_data1)
diff2 = np.abs(psd2 - fft_data2)
diff3 = np.abs(psd3 - fft_data3)
bbox = dict(facecolor='white', alpha=1, edgecolor='black', linewidth=0.5)

ax[1, 0].plot(w1, diff1, color=colors[1], label=f"AR Model w/ {order1}th\nOrder Yule-Walker",
                lw=3, zorder=2)
ax[1, 0].axhline(np.mean(diff1), color=colors[7], ls="--", lw=2, label="Mean")
ax[1, 0].text(0.65, 0.05, f"Mean: ${np.mean(diff1):.2e}$", transform=ax[1, 0].transAxes, bbox=bbox, fontsize=8)
ax[1, 0].set_ylabel("Absolute Difference")
ax[1, 0].set_yscale("log")

ax[1, 1].plot(w2, diff2, color=colors[2], label=f"AR Model w/ {order2}th\nOrder Yule-Walker",
                lw=3, zorder=2)
ax[1, 1].axhline(np.mean(diff2), color=colors[7], ls="--", lw=2, label="Mean")
ax[1, 1].text(0.65, 0.05, f"Mean: ${np.mean(diff2):.2e}$", transform=ax[1, 1].transAxes, bbox=bbox, fontsize=8)
ax[1, 1].set_ylabel("Absolute Difference")
ax[1, 1].set_yscale("log")

ax[1, 2].plot(w3, diff3, color=colors[4], label=f"AR Model w/ {order3}th\nOrder Yule-Walker",
                lw=3, zorder=2)
ax[1, 2].axhline(np.mean(diff3), color=colors[7], ls="--", lw=2, label="Mean")
ax[1, 2].text(0.65, 0.05, f"Mean: ${np.mean(diff3):.2e}$", transform=ax[1, 2].transAxes, bbox=bbox, fontsize=8)
ax[1, 2].set_ylabel("Absolute Difference")
ax[1, 2].set_yscale("log")

# Plot the poles of the AR model



plt.savefig(f"./MaxEntropy/Images/fft-ar-model-compare.pdf", dpi=500)
plt.show()

