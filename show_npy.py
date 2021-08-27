import numpy as np

with open('feat.npy', 'rb') as f:
    feat = np.load(f)

print(feat)
