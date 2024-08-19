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

ax[1].set_title("Signal Forecast")
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
# ax[1].legend()

plt.show()
