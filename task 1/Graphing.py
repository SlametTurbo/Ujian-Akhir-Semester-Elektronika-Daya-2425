import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

path = "LowSideDriving_GateSourceVoltage.txt"
path0 = "LowSideDriving_DrainSourceVoltage.txt"
path1 = "LowSideDriving_DrainCurrent.txt"

df = pd.read_csv(path, sep=r"\s+", engine="python")
df0 = pd.read_csv(path0, sep=r"\s+", engine="python")
df1 = pd.read_csv(path1, sep=r"\s+", engine="python")

t = df["time"].to_numpy()  
vgs = df["V(n002)"].to_numpy()
vds = df0["V(n001)"].to_numpy()
ids = df1["I(R1)"].to_numpy()

plt.figure()
plt.plot(t, ids, label="Ids", color="blue")
plt.xlabel("Time")
plt.ylabel("Ids (A)")
plt.title("Drain Current (Ids)")
plt.grid(True)
plt.tight_layout()
plt.show()