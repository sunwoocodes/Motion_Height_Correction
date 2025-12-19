import os
import numpy as np
from glob import glob
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from utils.viser_test import PoseViser

PROCESSED_DIR = os.path.join(ROOT, "data", "processed")

# PROCESSED_DIR = "./data/processed"

def load_dataset():
    raw_list = sorted(glob(f"{PROCESSED_DIR}/*_raw.npy"))
    X_all, Y_all = [], []

    for rp in raw_list:
        tp = rp.replace("_raw.npy", "_target.npy")
        if not os.path.exists(tp):
            print("Skip (no target):", rp)
            continue
        
        raw = np.load(rp)      # (T,33,3)
        tgt = np.load(tp)      # (T,33,3)

        # (T,33,3) → (T,99)
        raw = raw.reshape(len(raw), -1)
        tgt = tgt.reshape(len(tgt), -1)

        X_all.append(raw)
        Y_all.append(tgt)

    X = np.concatenate(X_all, axis=0)
    Y = np.concatenate(Y_all, axis=0)

    print(f"Loaded dataset: X={X.shape}, Y={Y.shape}")
    return X, Y
