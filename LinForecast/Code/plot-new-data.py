"""
Plot the additional data we have to eventually forecast
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr
import scipy.fft as fft

# Use custom style
mpl.style.use("./ma-style.mplstyle")
colors = cmr.take_cmap_colors("cmr.tropical", 8, cmap_range=(0, 0.85))

months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
months = enumerate(months, 1)
months = {v: k for k, v in months}
m_converter = lambda s: dict(months)[s.lower()]
date_converter = lambda s: str(m_converter(''.join([char for char in s if char.isalpha()]))) + "-" + (''.join([char for char in s if char.isdigit()]))

# Load data
data_p1 = "./SuppliedData/borza.dat"
data_p2 = "./SuppliedData/luna.dat"
data_p3 = "./SuppliedData/Wolf_number.dat"
data1 = np.genfromtxt(data_p1)
y2, ra2, dec2 = np.genfromtxt(data_p2, skip_header=1, unpack=True, converters={0: date_converter})
y3, m3, data3 = np.genfromtxt(data_p3, unpack=True)
data2 = np.sqrt(ra2**2 + dec2**2)
# data2 = ra2
m3 = np.array([str(int(m)) for m in m3])
time3 = [np.datetime64(f"{int(y3[i])}-{'0'+ m3[i] if len(m3[i]) <= 1 else m3[i]}-01") for i in range(len(y3))]

if False:
    # y2 = month-day
    # Horrible Solution but Data is Wack
    c = 0
    time2 = []
    for i, dp in enumerate(y2):
        month, day = dp.split("-")
        if c != 5:
            year_mod, _ =  divmod(i, 365)
        else:
            year_mod, _ = divmod(i, 366)
        year = 2000 - (5 - year_mod)
        if len(month) == 1:
            month = "0" + month
        if len(day) == 1:
            day = "0" + day
        time2.append([year, month, day])
        c = year_mod

    time2 = np.array(time2)
    errors = np.array([730, 1095, 1460, 1825])  # Identified after the fact
    errors_plus = np.array([1826, 1828])
    for e in errors:
        time2[e] = [int(time2[e][0])-1, time2[e][1], time2[e][2]]
    for e in errors_plus:
        time2[e] = [int(time2[e][0])+1, time2[e][1], time2[e][2]]
    print(time2[errors])
    print(time2[errors-1])
    print(y2[errors])
    time2_dt = [np.datetime64(f"{time2[i, 0]}-{time2[i, 1]}-{time2[i, 2]}") for i in range(len(time2))]
    time2_dt = np.array(time2_dt)
    plt.plot(time2_dt)
    plt.show()
    np.save("./LinForecast/Data/time_luna.npy", time2)    
    np.save("./LinForecast/Data/time_luna_dt.npy", time2_dt)
    quit()

# Load the processed times arrrr
time2 = np.load("./LinForecast/Data/time_luna_dt.npy")

# Save Sanitized Data
np.savez("./LinForecast/Data/sanitized_data.npz", data1=data1, data2=data2, data3=data3, time2=time2, time3=time3)

sr1 = 1 / (data1[1] - data1[0])
sr2 = 365
sr3 = 12
freq1 = np.linspace(0, sr1 / 2, len(data1) // 2)
freq2 = np.linspace(0, sr2 / 2, len(data2) // 2)
freq3 = np.linspace(0, sr3 / 2, len(data3) // 2)
spec1 = np.abs(fft.fft(data1)[:len(data1)//2])**2
spec1 /= np.max(spec1)
spec1[0] = 0
spec2 = np.abs(fft.fft(data2)[:len(data2)//2])**2
spec2 /= np.max(spec2)
spec2[0] = 0
spec3 = np.abs(fft.fft(data3)[:len(data3)//2])**2
spec3 /= np.max(spec3)
spec3[0] = 0

# Plot the data
fig, ax = plt.subplots(3, 2, figsize=(12, 9), layout="compressed")
ax[0, 0].plot(data1, color=colors[0])
ax[0, 0].set_title(f"Signal: {data_p1.split('/')[-1]}")
ax[0, 0].set_ylabel("Amplitude")
ax[0, 0].set_xlabel("Samples [Ticks]")

ax[1, 0].plot(time2, data2, color=colors[1], label="RA")
# ax[1, 0].plot(time2, dec2, color=colors[6], label="DEC")
ax[1, 0].set_title(f"Signal: {data_p2.split('/')[-1]}")
ax[1, 0].set_ylabel("Amplitude")
ax[1, 0].set_xlabel("Dates")

ax[2, 0].plot(time3, data3, color=colors[2])
ax[2, 0].set_title(f"Signal: {data_p3.split('/')[-1]}")
ax[2, 0].set_ylabel("Amplitude")
ax[2, 0].set_xlabel("Years")

# Compute the FFT of the signals
ax[0, 1].plot(freq1, spec1, color=colors[0])
ax[0, 1].set_title(f"FFT: {data_p1.split('/')[-1]}")
ax[0, 1].set_ylabel("Power Spectrum Density")
ax[0, 1].set_xlabel("Frequency [Ticks$^{-1}$]")
ax[0, 1].set_yscale("log")
ax[0, 1].set_xscale("log")

ax[1, 1].plot(freq2, np.abs(fft.fft(ra2)[:len(ra2)//2])**2, color=colors[1])
ax[1, 1].set_title(f"FFT: {data_p2.split('/')[-1]}")
ax[1, 1].set_ylabel("Power Spectrum Density")
ax[1, 1].set_xlabel("Frequency [Days$^{-1}$]")
ax[1, 1].set_yscale("log")
ax[1, 1].set_xscale("log")

ax[2, 1].plot(freq3, np.abs(fft.fft(data3)[:len(data3)//2])**2, color=colors[2])
ax[2, 1].set_title(f"FFT: {data_p3.split('/')[-1]}")
ax[2, 1].set_ylabel("Power Spectrum Density")
ax[2, 1].set_xlabel("Frequency [Months$^{-1}$]")
ax[2, 1].set_yscale("log")
ax[2, 1].set_xscale("log")

plt.savefig("./LinForecast/Images/new-data.pdf", dpi=500)
plt.show()
