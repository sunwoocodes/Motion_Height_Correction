import numpy as np
import tensorflow as tf
import os
import glob
import pandas as pd
import sys

# 경로 설정
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 시각화 도구 임포트
from utils.viser_test import PoseViser

# -----------------------------------------------------------
# [수정 1] 모델 파일 경로를 정확히 지정 (.keras 파일)
# -----------------------------------------------------------
MODEL_FILE_PATH = os.path.join(ROOT, "experiments", "height_mlp_model_2", "best_model.keras")

# 데이터 경로 (전처리가 완료된 데이터 사용)
PROCESSED_DIR = os.path.join(ROOT, "data", "processed", "test")
OUTPUT_DIR = os.path.join(ROOT, "data", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_trained_model():
    print(f"Loading model from: {MODEL_FILE_PATH}")
    if not os.path.exists(MODEL_FILE_PATH):
        raise FileNotFoundError(f"❌ 모델 파일이 없습니다! 경로를 확인하세요: {MODEL_FILE_PATH}")
    return tf.keras.models.load_model(MODEL_FILE_PATH)

def refine_sequence(model, pose_path, traj_path):
    # 1. 데이터 로드 (전처리된 데이터)
    raw_pose = np.load(pose_path) # (T, 33, 3)
    raw_traj = np.load(traj_path) # (T, 4)
    
    T = raw_pose.shape[0]

    # 2. 입력 데이터 준비 (Pose 99 + Trajectory 4 = 103차원)
    # 마지막 프레임은 예측할 '다음'이 없으므로 제외하고 T-1개만 예측
    curr_pose = raw_pose[:-1].reshape(T-1, -1) # (T-1, 99)
    curr_traj = raw_traj[:-1]                  # (T-1, 4)
    
    # 합치기 -> (Batch, 103)
    X = np.concatenate([curr_pose, curr_traj], axis=1)

    # 3. 예측 (Inference)
    print(f"  > Predicting {T-1} frames...")
    pred_flat = model.predict(X, verbose=0) # (T-1, 99)
    
    # 4. 형태 복원 (T-1, 33, 3)
    pred_pose_3d = pred_flat.reshape(T-1, 33, 3)

    # 5. 길이 맞추기 (마지막 프레임은 원본 그대로 붙여서 T 길이 유지)
    last_frame = raw_pose[-1].reshape(1, 33, 3)
    final_pose = np.concatenate([pred_pose_3d, last_frame], axis=0) # (T, 33, 3)

    return raw_pose, final_pose

def apply_foot_locking(data):
    """
    발이 바닥에 닿아있을 때(Grounding) 좌표를 고정하여 미끄러짐 방지
    data shape: (T, 33, 3)
    """
    # BlazePose 발 관련 인덱스: 27,28(발목), 29,30(뒤꿈치), 31,32(발가락)
    foot_indices = [27, 28, 29, 30, 31, 32]
    
    T = data.shape[0]
    
    # [설정] 민감도 조절 (이 값을 조절해서 미끄러짐을 잡으세요)
    # VEL_THRESH: 이 속도보다 느리면 '멈춘 것'으로 간주 (너무 크면 발이 쩍쩍 달라붙음)
    VEL_THRESH = 0.015  
    # HEIGHT_THRESH: 바닥에서 이 높이 이내에 있어야 고정 (너무 크면 공중에서 멈춤)
    HEIGHT_THRESH = 0.08 

    # 첫 프레임은 제외하고 1부터 시작
    for t in range(1, T):
        for idx in foot_indices:
            curr_pos = data[t, idx]   # 현재 프레임 위치
            prev_pos = data[t-1, idx] # 이전 프레임 위치
            
            # 1. 이동 거리(속도) 계산 (XZ 평면 기준)
            dist = np.linalg.norm(curr_pos[[0, 2]] - prev_pos[[0, 2]])
            
            # 2. 높이 확인 (Y값은 이미 Grounding 보정된 상태라 가정)
            height = curr_pos[1]

            # 3. 조건: 속도가 느리고 & 바닥에 가까우면 -> 위치 고정
            if dist < VEL_THRESH and abs(height) < HEIGHT_THRESH:
                # X, Z 좌표를 이전 프레임과 똑같이 만듦 (Lock)
                data[t, idx, 0] = prev_pos[0] 
                data[t, idx, 2] = prev_pos[2]
                
                # (선택) Y축(높이)도 고정하고 싶으면 아래 주석 해제
                # data[t, idx, 1] = prev_pos[1] 

    print("  -> Foot Locking Applied (Anti-Sliding)")
    return data

def save_to_csv(data, filename):
    """
    (T, 33, 3) 데이터를 CSV로 저장
    기능: 발바닥 착지(Grounding) + 겹침 방지(Widen) + 유니티 스케일 조정
    """
    # ------------------------------------------------------------------
    # [1] 발바닥 기준점 찾기 (노이즈 무시하고 바닥 착지)
    # ------------------------------------------------------------------
    feet_indices = [29, 30, 31, 32] 
    all_feet_y = data[:, feet_indices, 1] 
    ground_level = np.percentile(all_feet_y, 1) # 하위 1%를 바닥으로 간주

    MANUAL_OFFSET = 0
    
    # 바닥으로 내리기
    data[:, :, 1] -= (ground_level + MANUAL_OFFSET)

    data = apply_foot_locking(data)

    # ------------------------------------------------------------------
    # [2] 스케일 및 겹침 방지 설정
    # ------------------------------------------------------------------
    UNITY_SCALE = 1.2   # 전체 크기
    WIDTH_FACTOR = 1.2   # 좌우 벌리기 (겹침 방지)
    DEPTH_FACTOR = 1.2   # 앞뒤 벌리기 (겹침 방지)

    T, nBones, _ = data.shape
    rows = []
    
    for t in range(T):
        for b in range(nBones):
            x, y, z = data[t, b]
            
            # 스케일 적용
            x *= UNITY_SCALE
            y *= UNITY_SCALE
            z *= UNITY_SCALE
            
            # 뼈대 벌리기
            x *= WIDTH_FACTOR
            z *= DEPTH_FACTOR
            
            rows.append([t, b, x, y, z, 1.0])
    
    df = pd.DataFrame(rows, columns=["frame", "landmark", "x", "y", "z", "visibility"])
    save_path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(save_path, index=False)
    print(f"  -> CSV Saved: {save_path}")

def main():
    # 1. 모델 로드
    try:
        model = load_trained_model()
    except Exception as e:
        print(e)
        return

    # 2. 데이터 찾기 (data/processed 폴더에서)
    # 원하는 파일만 찾으려면 아래 검색어를 수정하세요 (예: "현시각")
    TARGET_KEYWORD = "" 
    
    all_files = glob.glob(os.path.join(PROCESSED_DIR, "*_pose.npy"))
    pose_files = [f for f in all_files if TARGET_KEYWORD in f]

    if not pose_files:
        print(f"❌ '{PROCESSED_DIR}'에서 테스트할 데이터(_pose.npy)를 찾지 못했습니다.")
        print("💡 먼저 '02_process_height_dataset.py'를 실행하여 데이터를 전처리해주세요.")
        return

    print(f"Found {len(pose_files)} sequences.")

    # 3. 반복 처리
    for p_path in pose_files:
        t_path = p_path.replace("_pose.npy", "_trajectory.npy")
        
        # 궤적 파일이 없으면 스킵 (짝이 맞아야 함)
        if not os.path.exists(t_path):
            print(f"⚠️ 궤적 파일 없음 (스킵): {os.path.basename(t_path)}")
            continue

        base_name = os.path.basename(p_path).replace("_pose.npy", "")
        print(f"\nProcessing: {base_name}")

        # 예측 실행
        raw_pose, refined_pose = refine_sequence(model, p_path, t_path)

        # 결과 저장 (.npy)
        npy_out = os.path.join(OUTPUT_DIR, f"{base_name}_refined.npy")
        np.save(npy_out, refined_pose)

        # 결과 저장 (.csv)
        save_to_csv(refined_pose, f"reconverted_{base_name}.csv")

        # 4. 시각화 (Offset 1.0으로 떨어뜨려서 보여줌)
        print("Displaying in PoseViser...")
        vis = PoseViser(fps=30)
        vis.play_two_sequences(raw_pose, refined_pose, offset=1.0)
        

if __name__ == "__main__":
    main()