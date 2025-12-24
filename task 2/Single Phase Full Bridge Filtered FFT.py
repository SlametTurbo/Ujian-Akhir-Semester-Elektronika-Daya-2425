import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = "Single Phase Full Bridge Filtered.csv"

F0_HZ  = 60.0        
FSW_HZ = 10_000.0     

# ambil steady-state
N_CYCLES_STEADY = 20

# resampling
FS_LOW  = 20_000     
FS_HIGH = 200_000  

# plot/FFT windows
N_CYCLES_PLOT     = 1
N_CYCLES_FFT_LOW  = 20
N_CYCLES_FFT_HIGH = 20

# plot limits
LOW_FMAX_HZ = 200
SW_SPAN_HZ  = 25_000

TIME_COL = None
SIG_COL  = None

# =======================
# HELPERS
# =======================
def pick_time_col(df):
    for c in df.columns:
        if "time" in c.lower():
            return c
    return df.columns[0]

def pick_signal_col(df, time_col):
    candidates = []
    for c in df.columns:
        if c == time_col:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if np.isfinite(s).sum() > 10:
            candidates.append(c)
    if not candidates:
        raise ValueError("Tidak menemukan kolom numeric untuk sinyal.")

    best, best_score = None, -1
    for c in candidates:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        x = x[np.isfinite(x)]
        score = np.percentile(np.abs(x - np.mean(x)), 95)
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
    if N < 128:
        raise ValueError("Data terlalu pendek untuk FFT.")
    w = np.hanning(N)
    X = np.fft.rfft(x * w)
    f = np.fft.rfftfreq(N, d=1/fs)

    cg = np.mean(w)               
    mag = (2.0/(N*cg)) * np.abs(X) 
    mag[0] *= 0.5
    return f, mag

def fundamental_peak_from_fft(f, mag, f0):
    i0 = np.argmin(np.abs(f - f0))
    return f[i0], mag[i0]

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
m = np.isfinite(t) & np.isfinite(x)
t, x = t[m], x[m]

print(f"[INFO] File     : {CSV_PATH}")
print(f"[INFO] Time col : {TIME_COL}")
print(f"[INFO] Sig  col : {SIG_COL}")

# =======================
# CUT STEADY-STATE (last N cycles)
# =======================
T0 = 1.0 / F0_HZ
t_end = float(np.max(t))
t_start = t_end - N_CYCLES_STEADY * T0

mask = t >= t_start
t_s = t[mask]
x_s = x[mask]

print(f"[OK] Steady window: t >= {t_start:.6f} s  (t_end={t_end:.6f} s)")
print(f"[OK] Samples steady: {len(t_s)}")

# =======================
# PLOT 1 CYCLE
# =======================
t2 = float(np.max(t_s))
t1 = t2 - N_CYCLES_PLOT * T0
tp, xp = resample_uniform(t_s, x_s, FS_LOW, t1, t2)

plt.figure()
plt.plot(tp - t1, xp)
plt.xlabel("Time within 1 cycle (s)")
plt.ylabel("Vout_filtered (V)")
plt.title("Filtered Output Voltage - 1 Cycle (steady-state)")
plt.grid(True)

# =======================
# FFT LOW (0..LOW_FMAX_HZ)
# =======================
t2l = float(np.max(t_s))
t1l = t2l - N_CYCLES_FFT_LOW * T0
_, xlf = resample_uniform(t_s, x_s, FS_LOW, t1l, t2l)
f_low, mag_low = fft_mag_peak(xlf, FS_LOW)

plt.figure()
mask_low = (f_low >= 0) & (f_low <= LOW_FMAX_HZ)
plt.plot(f_low[mask_low], mag_low[mask_low])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (Vpeak)")
plt.title(f"FFT LOW (Filtered) - last {N_CYCLES_FFT_LOW} cycles, Fs={FS_LOW} Hz")
plt.grid(True)

f1, V1pk = fundamental_peak_from_fft(f_low, mag_low, F0_HZ)
print(f"[LOW] Fundamental: f≈{f1:.3f} Hz, |V1|≈{V1pk:.3f} Vpeak, Vrms≈{V1pk/np.sqrt(2):.3f} Vrms")

# =======================
# FFT SWITCHING BAND around 10 kHz
# =======================
t2h = float(np.max(t_s))
t1h = t2h - N_CYCLES_FFT_HIGH * T0
_, xhf = resample_uniform(t_s, x_s, FS_HIGH, t1h, t2h)
f_hi, mag_hi = fft_mag_peak(xhf, FS_HIGH)

fmin = max(0.0, FSW_HZ - SW_SPAN_HZ)
fmax = FSW_HZ + SW_SPAN_HZ
mask_hi = (f_hi >= fmin) & (f_hi <= fmax)

plt.figure()
plt.plot(f_hi[mask_hi], mag_hi[mask_hi])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (Vpeak)")
plt.title(f"FFT Switching Band (Filtered) around {FSW_HZ/1000:.1f} kHz (±{SW_SPAN_HZ/1000:.1f} kHz)")
plt.grid(True)

plt.show()

