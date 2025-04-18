# load_npy.py
import numpy as np

def load_features(path):
    arr = np.load(path)
    return arr.tolist()
