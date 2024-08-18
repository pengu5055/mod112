"""
Calculate spectra of atmospheric CO2 fluctuations
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
