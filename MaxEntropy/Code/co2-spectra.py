"""
Calculate spectra of atmospheric CO2 fluctuations
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr
from scipy.optimize import curve_fit
from yw import YuleWalker

def square_fit(x, a, b, c):
    return a*x**2 + b*x + c

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

# Detrend the data
popt, pcov = curve_fit(square_fit, years, sig)
trend = square_fit(years, *popt)
trend_sigma = np.sqrt(np.diag(pcov))
detrended = sig - trend

# Plot intermediate results
fig, ax = plt.subplots(1, 2, figsize=(12, 5), layout="compressed")

ax[0].plot(years, sig, color=colors[0], label="Original Data")
ax[0].plot(years, trend, color=colors[1], label="Trend", lw=2, ls="--", zorder=3, alpha=0.8)
bbox = dict(facecolor='white', edgecolor='black', linewidth=0.5)
textstr = f"Fit: $y = a x^2 + b x + c$\n$a = {popt[0]:.4f} \pm {trend_sigma[0]:.2e}$\n" + \
          f"$b = {popt[1]:.2f} \pm {trend_sigma[1]:.2e}$\n$c = {popt[2]:.2e} \pm {trend_sigma[2]:.2e}$"
ax[0].text(0.05, 0.95, textstr, transform=ax[0].transAxes, fontsize=10,
           verticalalignment='top', bbox=bbox)
ax[0].set_title("Original Data")
ax[0].set_xlabel("Year")
ax[0].set_ylabel("CO2 Concentration [ppm/v]")
ax[0].legend()

ax[1].plot(years, detrended, color=colors[2], label="Detrended Data")
ax[1].set_title("Detrended Data")
ax[1].set_xlabel("Year")
ax[1].set_ylabel("CO2 Concentration [ppm/v]")

plt.savefig(f"./MaxEntropy/Images/detrended-co2.pdf", dpi=500)
plt.show()
