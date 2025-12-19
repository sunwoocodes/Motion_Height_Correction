import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# 경로 설정
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

# ---------------------------------------------------------
# [설정] 테스트할 비디오 파일 경로를 여기에 넣으세요!
# ---------------------------------------------------------
VIDEO_PATH = r"C:\Users\SUNWOO\Desktop\AI\AI_ML_UnityProject\AI_ML_Python_Final\final_project\data\raw_videos\현시각_1위_댄스챌린지_#다영_#body.mp4"
# ---------------------------------------------------------

# 저장될 위치: 바로 테스트가 가능하도록 processed/test 폴더로 지정
OUTPUT_DIR = os.path.join(ROOT, "data", "processed", "test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# MediaPipe 설정
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def _normalize(v, eps=1e-8):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)

def process_pipeline(raw_seq):
    """
    Raw 데이터(T, 33, 3)를 받아서 모델 입력용(Pose, Traj)으로 변환하는 핵심 함수
    """
    # 1. Smoothing (간단한 이동평균)
    T = raw_seq.shape[0]
    processed = raw_seq.copy()
    window = 5
    if T > window:
        kernel = np.ones(window)/window
        for i in range(33):
            for j in range(3):
                processed[:, i, j] = np.convolve(processed[:, i, j], kernel, mode='same')
    
    # 2. Body Frame Transform (Local 변환 + Trajectory 추출)
    LHIP, RHIP = 23, 24
    LSHO, RSHO = 11, 12
    
    hip_center = np.mean(processed[:, [LHIP, RHIP], :], axis=1)
    sho_center = np.mean(processed[:, [LSHO, RSHO], :], axis=1)
    
    x_axis = _normalize(processed[:, LHIP, :] - processed[:, RHIP, :])
    y_axis_raw = _normalize(sho_center - hip_center)
    z_axis = _normalize(np.cross(x_axis, y_axis_raw))
    z_axis[:, 1] = 0 # Roll 제거
    z_axis = _normalize(z_axis)
    y_axis = _normalize(np.cross(z_axis, x_axis))
    
    # 회전 행렬
    R = np.stack([x_axis, y_axis, z_axis], axis=1) # (T, 3, 3)
    
    # Local Pose 변환
    p = processed - hip_center[:, None, :]
    local_seq = np.einsum("tji,tbj->tbi", R, p)
    
    # Trajectory 생성 (x, y, z, rotation_y)
    azimuth = np.arctan2(z_axis[:, 0], z_axis[:, 2])
    traj = np.zeros((T, 4), dtype=np.float32)
    traj[:, :3] = hip_center
    traj[:, 3] = azimuth
    
    # 3. Scale Normalization
    torso_len = np.mean(np.linalg.norm(sho_center - hip_center, axis=1))
    scale = float(torso_len) if torso_len > 1e-8 else 1.0
    
    local_seq /= scale
    traj[:, :3] /= scale # 위치만 나눔
    
    # 4. Height Correction (바닥점 0으로 맞추기)
    LFOOT, RFOOT = 31, 32
    all_feet_y = np.concatenate([local_seq[:, LFOOT, 1], local_seq[:, RFOOT, 1]])
    min_ground = np.percentile(all_feet_y, 1) # 하위 1%
    local_seq[:, :, 1] -= min_ground
    
    return local_seq.astype(np.float32), traj.astype(np.float32)

def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ 영상을 찾을 수 없습니다: {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎬 영상 로드됨: {os.path.basename(VIDEO_PATH)} ({total_frames} frames)")

    raw_points = []
    
    # 1. MediaPipe 추출
    pbar = tqdm(total=total_frames, desc="Extracting Raw Keypoints")
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(frame_rgb)
        
        if res.pose_world_landmarks:
            # (33, 3) 배열 만들기
            pts = [[lm.x, -lm.y, lm.z] for lm in res.pose_world_landmarks.landmark]
            raw_points.append(pts)
        pbar.update(1)
    
    cap.release()
    pbar.close()
    
    if not raw_points:
        print("❌ 포즈를 찾지 못했습니다.")
        return

    raw_np = np.array(raw_points, dtype=np.float32) # (T, 33, 3)
    print(f"✅ 추출 완료: {raw_np.shape}")

    # 2. 전처리 (Smoothing -> Local -> Scale -> Grounding)
    print("⚙️ 전처리 진행 중 (Processing)...")
    final_pose, final_traj = process_pipeline(raw_np)
    
    # 3. 저장 (processed/test 폴더에 바로 저장!)
    base_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    
    pose_path = os.path.join(OUTPUT_DIR, f"{base_name}_pose.npy")
    traj_path = os.path.join(OUTPUT_DIR, f"{base_name}_trajectory.npy")
    
    np.save(pose_path, final_pose)
    np.save(traj_path, final_traj)
    
    print("\n🎉 [완료] 테스트 준비가 끝났습니다!")
    print(f"📂 저장 경로: {OUTPUT_DIR}")
    print(f"👉 이제 'src/test.py'를 바로 실행하세요!")

if __name__ == "__main__":
    main()