from dataset import load_dataset
from build_model import build_mlp
import tensorflow as tf
import os 
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "experiments", "height_mlp_model")

def train():
    # 1) 데이터 로딩
    X, Y = load_dataset()

    # 2) 모델
    model = build_mlp(input_dim=X.shape[1], output_dim=Y.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
        metrics=["mae"]
    )

    # 3) 학습
    history = model.fit(
        X, Y,
        batch_size=128,
        epochs=50,
        validation_split=0.1,
        shuffle=True
    )

    # 4) 저장
    model.save(MODEL_PATH)
    print(f"Saved → {MODEL_PATH}")

if __name__ == "__main__":
    train()
