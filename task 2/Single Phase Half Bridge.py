import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =======================
# USER SETTINGS
# =======================
CSV_PATH = "Single Phase Half Bridge Inverter Output.csv"
TIME_COL = "Time / s"
SIG_COL  = "Vm1:Measured voltage"

F0_HZ = 60.0            # sine fundamental
FSW_HZ = 10_000.0       # switching

# Resampling rates
FS_LOW  = 20_000    
FS_HIGH = 1_000_000  

# FFT settings
N_CYCLES_PLOT = 1
N_CYCLES_FFT_LOW  = 20  
N_CYCLES_FFT_HIGH = 2  

MAX_LOW_FFT_HZ = 5_000

# zoom switching band
SW_ZOOM_CENTER = FSW_HZ
SW_ZOOM_SPAN   = 20_000 

# =======================
# HELPERS
# =======================
def resample_uniform(t, x, fs, t_start, t_end):
    """Resample/interp dari time-step variable ke grid uniform."""
    order = np.argsort(t)
    t = t[order]
    x = x[order]

    tu, idx = np.unique(t, return_index=True)
    xu = x[idx]

    dt = 1.0 / fs
    t_u = np.arange(t_start, t_end, dt)
    x_u = np.interp(t_u, tu, xu)
    return t_u, x_u

def fft_mag_peak(x, fs):
    """
    Single-sided FFT magnitude in Vpeak (approx) with Hann window.
    Good for melihat amplitudo komponen sinus.
    """
    x = x - np.mean(x)
    N = len(x)
    if N < 32:
        raise ValueError("Data terlalu pendek untuk FFT.")

    w = np.hanning(N)
    X = np.fft.rfft(x * w)
    f = np.fft.rfftfreq(N, d=1/fs)

    cg = np.mean(w) 
    mag = (2.0 / (N * cg)) * np.abs(X)
    mag[0] *= 0.5
    return f, mag

# =======================
# LOAD
# =======================
df = pd.read_csv(CSV_PATH)
t = df[TIME_COL].to_numpy(float)
v = df[SIG_COL].to_numpy(float)

T0 = 1.0 / F0_HZ
t_end = float(np.max(t))

# =======================
# 1 CYCLE PLOT (last cycle)
# =======================
t1 = t_end - N_CYCLES_PLOT * T0
t2 = t_end
tp, vp = resample_uniform(t, v, FS_LOW, t1, t2)

plt.figure()
plt.plot(tp - t1, vp)
plt.xlabel("Time within 1 cycle (s)")
plt.ylabel("Vout (V)")
plt.title(f"Output Voltage - {N_CYCLES_PLOT} Cycle (F0={F0_HZ:.0f} Hz)")
plt.grid(True)

# =======================
# LOW-FREQ FFT (harmonics, THD view)
# =======================
t1l = t_end - N_CYCLES_FFT_LOW * T0
t2l = t_end
tlf, vlf = resample_uniform(t, v, FS_LOW, t1l, t2l)
f_low, mag_low = fft_mag_peak(vlf, FS_LOW)

plt.figure()
mask = f_low <= MAX_LOW_FFT_HZ
plt.plot(f_low[mask], mag_low[mask])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (Vpeak)")
plt.title(f"FFT (Low) - last {N_CYCLES_FFT_LOW} cycles, Fs={FS_LOW} Hz")
plt.grid(True)

# =======================
# HIGH-FREQ FFT (switching band)
# =======================
t1h = t_end - N_CYCLES_FFT_HIGH * T0
t2h = t_end
thf, vhf = resample_uniform(t, v, FS_HIGH, t1h, t2h)
f_high, mag_high = fft_mag_peak(vhf, FS_HIGH)

fmin = SW_ZOOM_CENTER - SW_ZOOM_SPAN
fmax = SW_ZOOM_CENTER + SW_ZOOM_SPAN
mask2 = (f_high >= fmin) & (f_high <= fmax)

plt.figure()
plt.plot(f_high[mask2], mag_high[mask2])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (Vpeak)")
plt.title(f"FFT (Switching band) around {FSW_HZ/1000:.0f} kHz, Fs={FS_HIGH} Hz")
plt.grid(True)

plt.show()
