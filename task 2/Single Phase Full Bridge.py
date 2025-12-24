import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =======================
# USER SETTINGS
# =======================
CSV_PATH = "Single Phase Full Bridge Inverter Output.csv"

# Kalau kamu tahu nama kolomnya, isi di sini (contoh: "Vab", "Vm1:Measured voltage", dll)
SIG_COL = None          # None = auto-detect
TIME_COL = None         # None = auto-detect (biasanya "Time / s")

F0_HZ = 60.0            # sine fundamental
FSW_HZ = 10_000.0       # ganti sesuai carrier kamu (mis. 10e3 atau 100e3)

# Resampling rates
FS_LOW  = 20_000        # cukup untuk harmonik fundamental
FS_HIGH = 1_000_000     # untuk switching band (pastikan > 2*FSW_HZ)

# Window length
N_CYCLES_PLOT = 1
N_CYCLES_FFT_LOW  = 20
N_CYCLES_FFT_HIGH = 10   # kalau FSW 10 kHz, 10 cycle masih aman

# Plot limits
LOW_FMAX = 5_000
SW_SPAN  = 20_000        # plot f_sw ± span (Hz)

# =======================
# HELPERS
# =======================
def pick_time_col(df):
    # cari kolom yang mengandung "time"
    for c in df.columns:
        if "time" in c.lower():
            return c
    # fallback: kolom pertama
    return df.columns[0]

def pick_signal_col(df, time_col):
    # ambil kolom numeric selain time
    num_cols = []
    for c in df.columns:
        if c == time_col:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if np.isfinite(s).sum() > 0:
            num_cols.append(c)

    if not num_cols:
        raise ValueError("Tidak menemukan kolom numeric untuk sinyal tegangan.")

    # pilih yang amplitude-nya paling besar (umumnya output inverter)
    best = None
    best_score = -np.inf
    for c in num_cols:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        x = x[np.isfinite(x)]
        if len(x) < 10:
            continue
        score = np.nanpercentile(np.abs(x - np.nanmean(x)), 95)
        if score > best_score:
            best_score = score
            best = c
    return best

def resample_uniform(t, x, fs, t_start, t_end):
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
    x = x - np.mean(x)
    N = len(x)
    if N < 64:
        raise ValueError("Data terlalu pendek untuk FFT.")
    w = np.hanning(N)
    X = np.fft.rfft(x * w)
    f = np.fft.rfftfreq(N, d=1/fs)

    cg = np.mean(w)                 # coherent gain Hann
    mag = (2.0/(N*cg)) * np.abs(X)  # ~Vpeak
    mag[0] *= 0.5
    return f, mag

# =======================
# LOAD
# =======================
df = pd.read_csv(CSV_PATH)

if TIME_COL is None:
    TIME_COL = pick_time_col(df)

if SIG_COL is None:
    SIG_COL = pick_signal_col(df, TIME_COL)

t = pd.to_numeric(df[TIME_COL], errors="coerce").to_numpy(float)
x = pd.to_numeric(df[SIG_COL], errors="coerce").to_numpy(float)

# bersihkan NaN
m = np.isfinite(t) & np.isfinite(x)
t = t[m]
x = x[m]

print(f"[INFO] Time column  : {TIME_COL}")
print(f"[INFO] Signal column: {SIG_COL}")

T0 = 1.0 / F0_HZ
t_end = float(np.max(t))

# =======================
# 1-CYCLE PLOT
# =======================
t1 = t_end - N_CYCLES_PLOT * T0
t2 = t_end
tp, xp = resample_uniform(t, x, FS_LOW, t1, t2)

plt.figure()
plt.plot(tp - t1, xp)
plt.xlabel("Time within 1 cycle (s)")
plt.ylabel("Vout (V)")
plt.title(f"Full-Bridge Output - {N_CYCLES_PLOT} Cycle (F0={F0_HZ:.0f} Hz)")
plt.grid(True)

# =======================
# LOW FFT (0..LOW_FMAX)
# =======================
t1l = t_end - N_CYCLES_FFT_LOW * T0
tlf, xlf = resample_uniform(t, x, FS_LOW, t1l, t_end)
f_low, mag_low = fft_mag_peak(xlf, FS_LOW)

plt.figure()
mask = (f_low >= 0) & (f_low <= LOW_FMAX)
plt.plot(f_low[mask], mag_low[mask])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (Vpeak)")
plt.title(f"FFT (Low) - last {N_CYCLES_FFT_LOW} cycles, Fs={FS_LOW} Hz")
plt.grid(True)

# =======================
# SWITCHING-BAND FFT (f_sw ± span)
# =======================
t1h = t_end - N_CYCLES_FFT_HIGH * T0
thf, xhf = resample_uniform(t, x, FS_HIGH, t1h, t_end)
f_hi, mag_hi = fft_mag_peak(xhf, FS_HIGH)

fmin = max(0.0, FSW_HZ - SW_SPAN)
fmax = FSW_HZ + SW_SPAN
mask2 = (f_hi >= fmin) & (f_hi <= fmax)

plt.figure()
plt.plot(f_hi[mask2], mag_hi[mask2])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (Vpeak)")
plt.title(f"FFT (Switching band) around {FSW_HZ/1000:.1f} kHz, Fs={FS_HIGH} Hz")
plt.grid(True)

plt.show()
