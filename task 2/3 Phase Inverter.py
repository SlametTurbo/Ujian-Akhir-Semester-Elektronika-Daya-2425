import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =======================
# SETTINGS
# =======================
CSV_PATH = "3 Phase Inverter.csv"

F0_HZ  = 60.0
FSW_HZ = 10_000.0

N_CYCLES_STEADY = 20

FS_LOW  = 20_000
FS_HIGH = 200_000

LOW_FMAX_HZ = 250
SW_SPAN_HZ  = 25_000

# =======================
# HELPERS
# =======================
def pick_time_col(df):
    for c in df.columns:
        if "time" in c.lower():
            return c
    return df.columns[0]

def pick_phase_cols(df, time_col):
    nums = []
    for c in df.columns:
        if c == time_col:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if np.isfinite(s).sum() > 10:
            nums.append(c)

    amps = []
    for c in nums:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        x = x[np.isfinite(x)]
        amps.append((np.percentile(np.abs(x), 95), c))

    amps.sort(reverse=True)
    return [c for _, c in amps[:3]]

def resample_uniform(t, x, fs, t1, t2):
    order = np.argsort(t)
    t = t[order]
    x = x[order]
    tu, idx = np.unique(t, return_index=True)
    xu = x[idx]

    dt = 1/fs
    t_u = np.arange(t1, t2, dt)
    x_u = np.interp(t_u, tu, xu)
    return t_u, x_u

def fft_mag(x, fs):
    x = x - np.mean(x)
    N = len(x)
    w = np.hanning(N)
    X = np.fft.rfft(x*w)
    f = np.fft.rfftfreq(N, 1/fs)

    cg = np.mean(w)
    mag = (2/(N*cg))*np.abs(X)
    mag[0] *= 0.5
    return f, mag

# =======================
# LOAD DATA
# =======================
df = pd.read_csv(CSV_PATH)

time_col = pick_time_col(df)
phase_cols = pick_phase_cols(df, time_col)

t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(float)
V = [pd.to_numeric(df[c], errors="coerce").to_numpy(float) for c in phase_cols]

mask = np.isfinite(t)
for i in range(3):
    mask &= np.isfinite(V[i])

t = t[mask]
V = [v[mask] for v in V]

print("[INFO] Time column :", time_col)
print("[INFO] Phase cols :", phase_cols)

# =======================
# CUT STEADY-STATE
# =======================
T0 = 1/F0_HZ
t_end = t.max()
t_start = t_end - N_CYCLES_STEADY*T0

idx = t >= t_start
t_s = t[idx]
V_s = [v[idx] for v in V]

# =======================
# 1 CYCLE TIME DOMAIN
# =======================
t2 = t_s.max()
t1 = t2 - T0

plt.figure()
for i, v in enumerate(V_s):
    tp, vp = resample_uniform(t_s, v, FS_LOW, t1, t2)
    plt.plot(tp-t1, vp, label=phase_cols[i])

plt.xlabel("Time within 1 cycle (s)")
plt.ylabel("Phase Voltage (V)")
plt.title("3-Phase Inverter Output (1 Cycle, Steady-State)")
plt.legend()
plt.grid(True)

# =======================
# FFT LOW (Phase A)
# =======================
_, vlf = resample_uniform(t_s, V_s[0], FS_LOW, t2-20*T0, t2)
f_low, mag_low = fft_mag(vlf, FS_LOW)

plt.figure()
m = (f_low >= 0) & (f_low <= LOW_FMAX_HZ)
plt.plot(f_low[m], mag_low[m])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (Vpeak)")
plt.title("FFT Low – Phase A (3-Phase Inverter)")
plt.grid(True)

# =======================
# FFT SWITCHING BAND
# =======================
_, vhf = resample_uniform(t_s, V_s[0], FS_HIGH, t2-20*T0, t2)
f_hi, mag_hi = fft_mag(vhf, FS_HIGH)

fmin = max(0, FSW_HZ-SW_SPAN_HZ)
fmax = FSW_HZ+SW_SPAN_HZ

plt.figure()
m = (f_hi >= fmin) & (f_hi <= fmax)
plt.plot(f_hi[m], mag_hi[m])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (Vpeak)")
plt.title("FFT Switching Band – Phase A")
plt.grid(True)

plt.show()
