"""
Detrend borza.dat with linear fit
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr
from scipy.optimize import curve_fit

# Use custom style
mpl.style.use("./ma-style.mplstyle")
colors = cmr.take_cmap_colors("cmr.tropical", 8, cmap_range=(0, 0.85))


load_path = "./SuppliedData/borza-detrended.npz"
with np.load("./LinForecast/Data/sanitized_data.npz") as d:
    y = range(len(d["data1"]))
    data = d["data1"]
    p0 = [1e10, -1e-10]
    def linear(x, a, b):
        return a*x + b
    popt, pcov = curve_fit(linear, y, data)
    yfit = linear(y, *popt)

fig, axs = plt.subplots(1, 2, figsize=(12, 5), layout="compressed")
ax = axs[0]
ax.plot(y, data, color=colors[2], label="Original Data")
ax.plot(y, yfit, color="black", label="Linear Fit", lw=2, ls="--")
ax.legend()
props = dict(facecolor="white", edgecolor="black", alpha=1)
textstr = f"Linear Fit: $y = ax + b$\n$a = {popt[0]:.2e} \pm {pcov[0, 0]**0.5:.2e}$" \
          + f"\n$b = {popt[1]:.2e} \pm {pcov[1, 1]**0.5:.2e}$"
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12, horizontalalignment="left",
        verticalalignment="top", bbox=props, fontweight="bold")
ax.set_title("Original Broker Data")
ax.set_xlabel("Sample")
ax.set_ylabel("Amplitude")

ax = axs[1]
ax.plot(y, data - yfit, color=colors[2])
ax.set_title("Detrended Broker Data")
ax.set_xlabel("Sample")
ax.set_ylabel("Amplitude")

plt.savefig(f"./LinForecast/Images/borza-detrended.pdf", dpi=500)
plt.show()
