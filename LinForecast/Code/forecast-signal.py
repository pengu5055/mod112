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
if False:
    load_path = "./SuppliedData/val3.dat"
    data = np.loadtxt(load_path)
    p0 = [1e3, -1e-6]
elif False:
    load_path = "./SuppliedData/co2-detrended.npz"
    with np.load("./MaxEntropy/Data/co2-detrended.npz") as data:
        y = data["years"]
        data = data["detrended"]
        p0 = [1e-3, -1e-6]
elif True:
    load_path = "./SuppliedData/borza-detrended.npz"
    with np.load("./LinForecast/Data/sanitized_data.npz") as d:
        y = range(len(d["data1"]))
        data = d["data1"]
        data -= data.mean()
        # y = d["time3"]
        p0 = [1e10, -1e-10]

# Split the data in half
print(len(data))
split = len(data) // 2
data1 = data[:split]
data2 = data[split:]

orders = np.array([4, 8, 6, 8, 10, 16, 24, 32]) * 10
fit_par = []
fit_cov = []
pred = []
peaks = []
for o in orders:
    yw = YuleWalker(data1, o)
    data_with_p = np.copy(data1)
    predictions = []
    for i in range(len(data2)):
        prediction = yw.predict_next_value(data_with_p)
        predictions.append(prediction)
        data_with_p = np.append(data_with_p, prediction)
    p_peaks = signal.find_peaks(predictions, height=0.01*np.max(data))[0]
    p_values = np.array([predictions[i] for i in p_peaks])
    p_peaks += split
    popt, pcov = curve_fit(exp, p_peaks, p_values, p0=p0)
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
b_fit_x = np.linspace(split, len(data), 100)
b_fit = exp(b_fit_x, *popt)

plt.scatter(b_peaks, b_values, color=colors[7], label="Positive Peaks", s=10)
plt.plot(b_fit_x, b_fit, color="black", lw=2, ls="--", label="Exponential Fit", alpha=0.8)
plt.show()

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

ax[0].plot([], [], color=colors[6], label="Envelope Max Amplitude")
ax[0].fill_between([], [], color=colors[6], alpha=0.2, label="Max Amplitude Std. Dev.")
ax[0].plot([], [], color=colors[1], label="Decay Rate", zorder=5)
ax[0].fill_between([], [], color=colors[1], alpha=0.2, zorder=5, label="Decay Rate Std. Dev.")
ax[0].axis('off')
ax[0].grid(False)
ax[0].set_facecolor('none')
ax[0].legend(loc="lower center", fontsize=10, ncols=2)


# Signal View and Fit
print(len(data1), len(data2))
print(np.array(b_predictions).shape)
print(np.array(x_forecast_axis).shape)
ax[1].plot(data1, color=colors[0], label="Input Data")
ax[1].plot(x_forecast_axis, b_predictions, color=colors[1], label="Forecast")
ax[1].scatter(b_peaks, b_values, color=colors[7], label="Positive Peaks", s=10)
ax[1].plot(b_fit_x, b_fit, color="black", lw=2, ls="--", label="Exponential Fit", alpha=0.8)
ax[1].plot(b_fit_x, -b_fit, color="black", lw=2, ls="--", alpha=0.8)
props = dict(facecolor='white', alpha=1, edgecolor="black", lw=0.5)
textstr = f"Exponential Fit: $y = a \cdot e^{{b \cdot x}}$\n$a = {popt[0]:.2e} \pm {pcov[0, 0]**2:.2e}$" \
          f"\n$b = {popt[1]:.2e} \pm {pcov[1, 1]**2:.2e}$"
ax[1].text(0.97, 0.04, textstr, transform=ax[1].transAxes, fontsize=10, fontweight="bold",
            verticalalignment="bottom", horizontalalignment="right", bbox=props)
# ax[1].legend()

# Abs. diff from data2
for i, p in enumerate(pred):
    if i == b_max:
        continue
    ax[2].plot(np.abs(data2 - p), alpha=0.8, color=colors[i])
ax[2].plot(np.abs(data2 - b_predictions), alpha=1, color=colors[b_max], zorder=5, lw=1.5)

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
props = dict(facecolor='white', alpha=1, edgecolor="black", lw=0.5)
ax[3].text(0.6, 0.5, "Makes no sense to fit a\ndecay envelope to the data", transform=ax[3].transAxes, fontsize=12,
              verticalalignment="center", horizontalalignment="center", bbox=props)

plt.savefig(f"./LinForecast/Images/forecast-{load_path.split('/')[-1]}.pdf", dpi=500)
plt.show()
