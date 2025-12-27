import tensorflow as tf
from tensorflow.keras import layers, models

def build_mlp(input_dim, output_dim):
    """
    [업그레이드 버전] 
    - BatchNormalization 추가: 학습 안정화 및 가속
    - Dropout 추가: 과적합 방지
    - Residual Connection (Optional): 잔차 학습 유도
    """
    
    # 입력층
    inputs = layers.Input(shape=(input_dim,))
    
    # -----------------------------------------------------
    # Hidden Layer 1: 충분한 정보 추출을 위해 뉴런 수 512로 증가
    # -----------------------------------------------------
    x = layers.Dense(512)(inputs)
    x = layers.BatchNormalization()(x) # 활성화 함수 전에 적용하는 것이 정석
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)         # 20%의 뉴런을 랜덤하게 끔

    # -----------------------------------------------------
    # Hidden Layer 2
    # -----------------------------------------------------
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)

    # -----------------------------------------------------
    # Hidden Layer 3: 깊이를 하나 더 추가해 복잡한 패턴 학습
    # -----------------------------------------------------
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)

    # -----------------------------------------------------
    # 출력층 (Regression)
    # -----------------------------------------------------
    outputs = layers.Dense(output_dim)(x)
    
    # [Pro Tip] Residual Connection (잔차 연결)
    # 만약 input_dim과 output_dim이 같다면(예: Pose -> Pose 보정), 
    # 원본 입력을 출력에 더해주는 것이 학습에 매우 유리합니다.
    # 여기서는 input(103) -> output(99)로 차원이 달라서 단순 더하기는 불가능하므로
    # 위와 같이 기본적인 MLP 구조로 마무리.

    model = models.Model(inputs=inputs, outputs=outputs, name="Motion_MLP_V2")
    return model