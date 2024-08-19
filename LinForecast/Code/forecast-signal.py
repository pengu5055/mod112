"""
Forecast signal by giving half of the signal to the Yule-Walker method,
and predict the next half of the signal.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr
from yw import YuleWalker
from scipy.optimize import curve_fit
import scipy.signal as signal

def exp(x, a, b):
    return a * np.exp(b * x)

# Use custom style
mpl.style.use("./ma-style.mplstyle")
colors = cmr.take_cmap_colors("cmr.tropical", 8, cmap_range=(0, 0.85))

# Load data
load_path = "./SuppliedData/val2.dat"
data = np.loadtxt(load_path)

# Split the data in half
split = len(data) // 2
data1 = data[:split]
data2 = data[split:]

# Forecast the second half of the signal
order = 5
yw = YuleWalker(data1, order)
data_with_p = np.copy(data1)
predictions = []
for i in range(256):
    prediction = yw.predict_next_value(data_with_p)
    predictions.append(prediction)
    data_with_p = np.append(data_with_p, prediction)

# Plot the data and the forecast
fig, ax = plt.subplots(2, 2, figsize=(12, 9), layout="compressed")
ax = ax.flatten()
x_input_axis = range(0, split)
x_forecast_axis = range(split, len(data))

ax[0].set_visible(True)
ax[0].plot([], [], color=colors[0], label="Input Data")
ax[0].plot([], [], color=colors[1], label="Forecast")
ax[0].scatter([], [], color=colors[7], label="Positive Peaks", s=10)
ax[0].plot([], [], color="black", lw=1, ls="--", label="Exponential Fit", alpha=0.8)
ax[0].legend(loc="center", fontsize=12)
textstr = f"Forecasted Signal {load_path.split('/')[-1]}\n{order}th Order AR Model"
ax[0].text(0.5, 0.8, textstr, transform=ax[0].transAxes, fontsize=18, fontweight="bold",
            verticalalignment="center", horizontalalignment="center")
ax[0].axis('off')
ax[0].grid(False)
ax[0].set_facecolor('none')



ax[1].plot(data1, color=colors[0], label="Input Data")
ax[1].plot(x_forecast_axis, predictions, color=colors[1], label="Forecast")
pred_peaks = signal.find_peaks(predictions, height=0.02)[0]
peak_values = np.array([predictions[i] for i in pred_peaks])
pred_peaks += split
ax[1].scatter(pred_peaks, peak_values, color=colors[7], label="Positive Peaks", s=10)
popt, pcov = curve_fit(exp, pred_peaks, peak_values, p0=[1000, -1e-5])
y_fit = exp(pred_peaks, *popt)
ax[1].plot(pred_peaks, y_fit, color="black", lw=1, ls="--", label="Exponential Fit", alpha=0.8)
ax[1].plot(pred_peaks, -y_fit, color="black", lw=1, ls="--", alpha=0.8)
props = dict(facecolor='white', alpha=1, edgecolor="black", lw=0.5)
textstr = f"Exponential Fit: $y = a \cdot e^{{b \cdot x}}$\n$a = {popt[0]:.2e} \pm {pcov[0, 0]**2:.2e}$" \
          f"\n$b = {popt[1]:.2e} \pm {pcov[1, 1]**2:.2e}$"
ax[1].text(0.97, 0.96, textstr, transform=ax[1].transAxes, fontsize=10, fontweight="bold",
            verticalalignment="top", horizontalalignment="right", bbox=props)
# ax[1].legend()

plt.show()
