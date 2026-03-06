import numpy as np
import os

SRC = "/Users/lilykoffman/Documents/ssl-wearables/capture24_100hz_w10_o0"
OUT = "/Users/lilykoffman/Documents/ssl-wearables/capture24_100hz_w10_o0_small"

os.makedirs(OUT, exist_ok=True)

X = np.load(os.path.join(SRC, "X.npy"), mmap_mode="r")
Y = np.load(os.path.join(SRC, "Y.npy"), allow_pickle=True)
P = np.load(os.path.join(SRC, "pid.npy"), allow_pickle=True)

rng = np.random.RandomState(0)
n = len(Y)
k = 20000  # try 5k–50k for dev
idx = rng.choice(n, size=k, replace=False)

np.save(os.path.join(OUT, "X.npy"), X[idx])
np.save(os.path.join(OUT, "Y.npy"), Y[idx])
np.save(os.path.join(OUT, "pid.npy"), P[idx])

print("Saved subset to", OUT, "with", k, "windows")