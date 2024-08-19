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

orders = [2, 4, 8, 6, 8, 10, 16, 24]
fit_par = []
fit_cov = []
pred = []
peaks = []
for o in orders:
    yw = YuleWalker(data1, o)
    data_with_p = np.copy(data1)
    predictions = []
    for i in range(256):
        prediction = yw.predict_next_value(data_with_p)
        predictions.append(prediction)
        data_with_p = np.append(data_with_p, prediction)
    p_peaks = signal.find_peaks(predictions, height=0.02)[0]
    p_values = np.array([predictions[i] for i in p_peaks])
    p_peaks += split
    popt, pcov = curve_fit(exp, p_peaks, p_values, p0=[1e10, -1e-2])
    fit_par.append(popt)
    fit_cov.append(pcov)
    pred.append(predictions)
    peaks.append([p_peaks, p_values])

# Find largest b value
b_max = np.array(fit_par)[:, 1].argmax()
b_peaks, b_values = peaks[b_max]
b_order = orders[b_max]
b_predictions = pred[b_max]
popt = fit_par[b_max]
b_fit_x = np.linspace(b_peaks.min(), b_peaks.max(), 100)
b_fit = exp(b_fit_x, *popt)

# Plot the data and the forecast
fig, ax = plt.subplots(2, 2, figsize=(12, 9), layout="compressed")
ax = ax.flatten()
x_input_axis = range(0, split)
x_forecast_axis = range(split, len(data))

ax[0].set_visible(True)
ax[0].plot([], [], color=colors[0], label="Input Data")
ax[0].plot([], [], color=colors[1], label=f"{b_order}th Order (Best Forecast)")
ax[0].scatter([], [], color=colors[7], label="Positive Peaks", s=10)
ax[0].plot([], [], color="black", lw=1, ls="--", label="Exponential Fit", alpha=0.8)

textstr = f"Forecasted Signal {load_path.split('/')[-1]}"
ax[0].text(0.5, 0.7, textstr, transform=ax[0].transAxes, fontsize=18, fontweight="bold",
            verticalalignment="center", horizontalalignment="center")
ax[0].text(0.5, 0.66, f"\nBest AR Model was {b_order}th Order", transform=ax[0].transAxes, fontsize=14,
            verticalalignment="center", horizontalalignment="center")
for i, p in enumerate(pred):
    ax[0].plot([], [], alpha=0.5, color=colors[i], label=f"Abs. Err. {orders[i]}th Order")

ax[0].plot([], [], color=colors[6], label="Amplitude")
ax[0].fill_between([], [], color=colors[6], alpha=0.2, label="Amplitude Std. Dev.")
ax[0].plot([], [], color=colors[1], label="Decay Rate", zorder=5)
ax[0].fill_between([], [], color=colors[1], alpha=0.2, zorder=5, label="Decay Rate Std. Dev.")
ax[0].axis('off')
ax[0].grid(False)
ax[0].set_facecolor('none')
ax[0].legend(loc="lower center", fontsize=10, ncols=2)


# Signal View and Fit
ax[1].plot(data1, color=colors[0], label="Input Data")
ax[1].plot(x_forecast_axis, b_predictions, color=colors[1], label="Forecast")
ax[1].scatter(b_peaks, b_values, color=colors[7], label="Positive Peaks", s=10)
ax[1].plot(b_fit_x, b_fit, color="black", lw=2, ls="--", label="Exponential Fit", alpha=0.8)
ax[1].plot(b_fit_x, -b_fit, color="black", lw=2, ls="--", alpha=0.8)
props = dict(facecolor='white', alpha=1, edgecolor="black", lw=0.5)
textstr = f"Exponential Fit: $y = a \cdot e^{{b \cdot x}}$\n$a = {popt[0]:.2e} \pm {pcov[0, 0]**2:.2e}$" \
          f"\n$b = {popt[1]:.2e} \pm {pcov[1, 1]**2:.2e}$"
ax[1].text(0.97, 0.96, textstr, transform=ax[1].transAxes, fontsize=10, fontweight="bold",
            verticalalignment="top", horizontalalignment="right", bbox=props)
# ax[1].legend()

# Abs. diff from data2
for i, p in enumerate(pred):
    ax[2].plot(np.abs(data2 - p), alpha=0.8, color=colors[i])

ax[2].set_yscale("log")
ax[2].set_ylabel("Abs. Diff. from Second Half of Signal")
ax[2].set_xlabel("Sample")
ax[2].set_title("Abs. Diff. from Second Half of Signal")



# Plot Fit Parameters
ax2 = ax[3].twinx()
ax[3].set_title("Signal Decay Parameters")
ax[3].set_xlabel("Order")
ax[3].set_ylabel("Decay Rate")
# Order data monotonically
orders = np.array(orders)
fit_par = np.array(fit_par)
fit_cov = np.array(fit_cov)
order_sort = np.argsort(orders)
orders = orders[order_sort]
fit_par = fit_par[order_sort]
fit_cov = fit_cov[order_sort]
a_std = np.array(fit_cov)[:, 0, 0]**0.5
ax2.plot(orders, np.array(fit_par)[:, 0], color=colors[6], label="Amplitude")
ax2.fill_between(orders, np.array(fit_par)[:, 0] - a_std, np.array(fit_par)[:, 0] + a_std, color=colors[6], alpha=0.2)
ax2.set_yscale("log")
ax2.set_ylabel("Decay Envelope Max Amplitude")
ax[3].set_xticks(orders)
ax[3].plot(orders, np.array(fit_par)[:, 1], color=colors[1], label="Decay Rate", zorder=5)
std = np.array(fit_cov)[:, 1, 1]**0.5
ax[3].fill_between(orders, np.array(fit_par)[:, 1] - std, np.array(fit_par)[:, 1] + std, 
                   color=colors[1], alpha=0.2, zorder=5)

plt.savefig(f"./LinForecast/Images/forecast-{load_path.split('/')[-1]}.pdf", dpi=500)
plt.show()
