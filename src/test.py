import numpy as np
import tensorflow as tf
import os
import glob
import pandas as pd
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from utils.viser_test import PoseViser

MODEL_PATH = os.path.join(ROOT, "experiments", "height_mlp_model")
# MODEL_PATH = "experiments/height_mlp_model"

def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def refine_sequence(raw_path, out_path=None):
    model = load_model()

    raw = np.load(raw_path)    # (T,33,3)
    T = raw.shape[0]

    x = raw.reshape(T, -1)     # (T,99)
    y = model.predict(x)

    refined = y.reshape(T, 33, 3)

    if out_path is not None:
        np.save(out_path, refined)
        print(f"Saved refined → {out_path}")

    return refined

def main():
    TEST_DATA_PATH = os.path.join(ROOT, "data", "test_keypoints")
    for raw_path in glob.glob(f"{TEST_DATA_PATH}/*_raw.npy"):
        out_path = raw_path.replace("_raw.npy", "_refined.npy")
        refine_sequence(raw_path, out_path)
    
        raw = np.load(raw_path)
        refine = np.load(out_path)
        
        T, nBones,_ = refine.shape
        rows = []
        for t in range(T):
            for b in range(nBones):
                x,y,z = refine[t,b]
                rows.append([t,b,x,y,z,1.0])
        
        df = pd.DataFrame(rows, columns=["frame","landmark","x","y","z","visibility"])
        filename = os.path.basename(raw_path)      # test_raw.npy
        name_only = os.path.splitext(filename)[0]   # test_raw

        TEST_OUTPUT_PATH = os.path.join(ROOT,"data", "output")
        df.to_csv(f"{TEST_OUTPUT_PATH}/reconverted_{name_only}.csv",index=False)
        print("test CSV saved!")

    # visualize in web viewer
    vis = PoseViser(fps=30)
    vis.play_two_sequences(raw,refine,offset=0.0)

    

if __name__ == "__main__":
    main()