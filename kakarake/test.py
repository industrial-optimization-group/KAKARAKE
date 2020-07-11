import numpy as np
import pandas as pd
data = pd.read_csv("c432-110.csv", header=None, sep="\s+")[range(2, 11)]
positions = np.linspace(0, 1, len(data.columns))
od = [6, 1, 5, 4, 0, 8, 2, 7, 3]

a = np.ones_like(od, dtype=float)

for i, val in zip(od, positions):
    a[i] = val

print(a)