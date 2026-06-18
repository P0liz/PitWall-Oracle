import pandas as pd
import numpy as np
import os, time

print("generating dataframe")
times = []
drivers = []
is_valid = []
for i in range(1*10**7):
    times.append(np.random.randint(1, 2) + np.random.random())
    drivers.append(np.random.choice(["hamilton", "button", "vettel", "antonelli", "leclec", "verstappen"], size=1))
    is_valid.append(np.random.random() > 0.5)
d = {"lap_time": times, "driver": drivers, "is_valid": is_valid}
df = pd.DataFrame(data=d)
print("dataframe created")

print("writing csv file")
df.to_csv("f1.csv")
print("writing parquet file")
df.to_parquet("f1.parquet")

for fname in ["f1.csv", "f1.parquet"]:
    size_mb = os.path.getsize(fname) / (1024 ** 2)
    t0 = time.perf_counter()
    pd.read_csv(fname) if fname.endswith(".csv") else pd.read_parquet(fname)
    elapsed = time.perf_counter() - t0
    print(f"{fname}: {size_mb:.2f} MB, read in {elapsed:.3f}s")
