import pandas as pd
import matplotlib.pyplot as plt

# Path file (ubah sesuai lokasi di PC kamu)
path = "Totem-Pole Driving.txt"

# Baca file LTspice
df = pd.read_csv(path, sep=r"\s+", engine="python")

# Ambil data
t = df["time"].to_numpy() * 1e6      # waktu → µs
v_n002 = df["V(n002)"].to_numpy()    # Tegangan node n002
v_n003 = df["V(n003)"].to_numpy()    # Tegangan node n003 (Vgs)
i_load = df["I(Rload)"].to_numpy()   # Arus beban

plt.figure()
plt.plot(t, v_n002)
plt.title("Vds")
plt.xlabel("Time [µs]")
plt.ylabel("Voltage [V]")
plt.grid(True)

plt.figure()
plt.plot(t, v_n003)
plt.title("Vgs")
plt.xlabel("Time [µs]")
plt.ylabel("Voltage [V]")
plt.grid(True)

plt.figure()
plt.plot(t, i_load)
plt.title("Id")
plt.xlabel("Time [µs]")
plt.ylabel("Current [A]")
plt.grid(True)

plt.show()

