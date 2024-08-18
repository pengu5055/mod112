"""
Just plot the given data.
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
years, sig = np.genfromtxt("./SuppliedData/co2_ex.dat", unpack=True)

# Replace missing values with closest valid value (-99.99 indicates missing)
sig[sig == -99.99] = np.nan
nan_indices = np.argwhere(np.isnan(sig)).flatten()

# Replace missing values with the closest valid value
for i in nan_indices:
    valid_indices = np.argwhere(~np.isnan(sig)).flatten()
    closest_index = valid_indices[np.argmin(np.abs(valid_indices - i))]
    sig[i] = sig[closest_index]

data1 = np.genfromtxt("./SuppliedData/val2.dat")
data2 = np.genfromtxt("./SuppliedData/val3.dat")

# Plot the supplied data
fig, ax = plt.subplots(1, 3, figsize=(12, 5), layout="compressed")

ax[0].plot(data1, color=colors[0])
ax[0].set_title(f"Signal val2.dat")
ax[0].set_xlabel("Sample")
ax[0].set_ylabel("Amplitude")

ax[1].plot(data2, color=colors[1])
ax[1].set_title(f"Signal val3.dat")
ax[1].set_xlabel("Sample")
ax[1].set_ylabel("Amplitude")

ax[2].plot(years, sig, color=colors[3])
ax[2].set_title("Atmospheric CO2 from co2.dat")
ax[2].set_xlabel("Year")
ax[2].set_ylabel("CO2 Concentration [ppm/v]")
ax[2].set_xlim(years[0], years[-1])

plt.savefig(f"./MaxEntropy/Images/supplied-data.pdf", dpi=500)
plt.show()
