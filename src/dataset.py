import os
import numpy as np
from glob import glob
import sys

# 경로 설정
# 현재 파일(dataset.py)이 src 폴더에 있다면, 두 번 올라가야 프로젝트 루트(final_project)입니다.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")

def load_dataset():
    """
    학습용 데이터(X, Y)를 로드합니다.
    """
    # 1. 디버깅: 경로가 맞는지 먼저 확인
    if not os.path.exists(PROCESSED_DIR):
        print(f"❌ [Error] 데이터 폴더가 없습니다: {PROCESSED_DIR}")
        print("   -> 경로 설정(ROOT)을 확인해보세요.")
        return np.array([]), np.array([])

    # 2. 파일 찾기 (_pose.npy 로 찾아야 함!)
    pose_files = sorted(glob(os.path.join(PROCESSED_DIR, "*_pose.npy")))
    print(f"[Debug] 검색 경로: {PROCESSED_DIR}")
    print(f"[Debug] 발견된 파일 수: {len(pose_files)}")

    if len(pose_files) == 0:
        raise ValueError(f"❌ '{PROCESSED_DIR}' 경로에 '_pose.npy' 파일이 하나도 없습니다! 전처리가 잘 됐는지 확인하세요.")
    
    X_all, Y_all = [], []

    for p_path in pose_files:
        # 짝이 되는 Trajectory 파일 경로 찾기
        t_path = p_path.replace("_pose.npy", "_trajectory.npy")
        
        if not os.path.exists(t_path):
            print(f"Skipping {os.path.basename(p_path)}: Trajectory file missing.")
            continue
            
        # 3. 데이터 로드
        pose = np.load(p_path) # (T, 33, 3)
        traj = np.load(t_path) # (T, 4)

        # 포즈 평탄화: (T, 33, 3) -> (T, 99)
        N = pose.shape[0]
        pose_flat = pose.reshape(N, -1)
        
        # 4. 데이터 짝 맞추기 (Next Frame Prediction)
        # X: 현재 [Pose + Traj]
        # Y: 다음 [Pose]
        curr_pose = pose_flat[:-1]      # 0 ~ T-2
        curr_traj = traj[:-1]           # 0 ~ T-2
        next_pose = pose_flat[1:]       # 1 ~ T-1
        
        # 합치기
        X_data = np.concatenate([curr_pose, curr_traj], axis=1)
        Y_data = next_pose
        
        X_all.append(X_data)
        Y_all.append(Y_data)

    # 전체 데이터 병합
    X = np.concatenate(X_all, axis=0)
    Y = np.concatenate(Y_all, axis=0)

    print(f"✅ Dataset Loaded: X={X.shape}, Y={Y.shape}")
    return X, Y