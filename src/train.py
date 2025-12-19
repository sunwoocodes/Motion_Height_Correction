import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from dataset import load_dataset
from build_model import build_mlp

# -----------------------------------------------------------
# [안전장치] GPU 메모리 점유율 제한 (OOM 에러 방지)
# -----------------------------------------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 경로 설정
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENT_DIR = os.path.join(ROOT, "experiments", "height_mlp_model_2")
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

# 저장할 모델 파일 경로 (.keras 권장)
MODEL_PATH = os.path.join(EXPERIMENT_DIR, "best_model.keras")

def train():
    # 1) 데이터 로딩
    print("\n[1] Loading Dataset...")
    try:
        X, Y = load_dataset()
        print(f"   -> X shape: {X.shape}, Y shape: {Y.shape}")
    except Exception as e:
        print(f"❌ 데이터 로딩 실패: {e}")
        return

    # 2) 모델 빌드
    print("\n[2] Building Model...")
    model = build_mlp(input_dim=X.shape[1], output_dim=Y.shape[1])
    
    # 컴파일 (Adam 옵티마이저)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    model.summary()

    # -----------------------------------------------------------
    # [핵심] 콜백 설정: 가장 똑똑할 때 저장하고, 더 안 늘면 그만두기
    # -----------------------------------------------------------
    callbacks = [
        # 검증 손실(val_loss)이 가장 낮을 때만 저장
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        # 15번 동안 성능 향상이 없으면 조기 종료
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True
        )
    ]

    # 3) 학습 시작
    print("\n[3] Start Training...")
    history = model.fit(
        X, Y,
        batch_size=64,         # 데이터가 적을 땐 128보다 64가 안정적
        epochs=100,            # EarlyStopping 믿고 넉넉하게 100 설정
        validation_split=0.2,  # 검증 데이터 20% 사용
        shuffle=True,
        callbacks=callbacks
    )

    print(f"\n✅ Training Finished. Best model saved at: {MODEL_PATH}")

if __name__ == "__main__":
    train()