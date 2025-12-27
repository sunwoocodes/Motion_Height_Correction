import os
import numpy as np
import pandas as pd
from glob import glob
import sys
import traceback
import time

# 스레드 꼬임 방지
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RAW_KEYPOINTS_DIR = os.path.join(ROOT, "data", "raw_keypoints")
OUT_DIR = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)


def load_raw_keypoints(npz_path):
    # npz 파일 로드 시 예외처리 추가
    try:
        data = np.load(npz_path, allow_pickle=True)["data"]
    except KeyError:
        # 혹시 키가 다를 경우를 대비해 keys 확인
        f = np.load(npz_path, allow_pickle=True)
        keys = list(f.keys())
        data = f[keys[0]]

    df = pd.DataFrame(data, columns=["frame", "landmark", "x", "y", "z", "visibility"])

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["z"] = pd.to_numeric(df["z"], errors="coerce")

    pts_seq = []
    # 프레임 순서 보장을 위해 정렬
    frames = sorted(df["frame"].unique())
    for f in frames:
        group = df[df["frame"] == f].sort_values("landmark")
        pts = group[["x", "y", "z"]].values
        pts_seq.append(pts)

    seq = np.array(pts_seq, dtype=np.float32)
    return seq


def smooth_pose_data(seq, window_length=9, polyorder=3):
    """
    Scipy 없이 이동평균(컨볼루션) 스무딩.
    """
    processed = seq.copy()
    T = processed.shape[0]
    # print(f"    [smooth] Using Pure Numpy Moving Average (T={T}, win={window_length})")

    if window_length % 2 == 0:
        window_length += 1
    if T <= window_length:
        return processed

    kernel = np.ones(window_length, dtype=np.float32) / window_length
    pad = window_length // 2

    for i in range(33):
        for j in range(3):
            data = processed[:, i, j]
            padded = np.pad(data, (pad, pad), mode="edge")
            smoothed = np.convolve(padded, kernel, mode="valid")
            processed[:, i, j] = smoothed[:T]

    return processed


def _normalize(v, eps=1e-8):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)


def process_body_frame_transform(seq):
    """
    포즈를 로컬 좌표계로 변환하고, 
    Trajectory(루트 위치 + 회전) 정보를 함께 반환.
    """
    T = seq.shape[0]
    LHIP, RHIP = 23, 24
    LSHO, RSHO = 11, 12

    # 1. 힙 센터(Root) 계산
    hip_center = np.mean(seq[:, [LHIP, RHIP], :], axis=1) # (T, 3)
    sho_center = np.mean(seq[:, [LSHO, RSHO], :], axis=1)

    # 2. 로컬 좌표계 축 생성
    # X축: 좌->우 힙
    x_axis = seq[:, LHIP, :] - seq[:, RHIP, :]
    x_axis = _normalize(x_axis)

    # 임시 Y축: 힙->어깨 (척추 방향)
    y_axis_raw = sho_center - hip_center
    y_axis_raw = _normalize(y_axis_raw)

    # Z축: 전방 (X와 Y의 외적)
    z_axis = np.cross(x_axis, y_axis_raw)
    z_axis = _normalize(z_axis)

    # Roll 제거: Z축을 수평면에 투영 (y성분 0으로 만듦)
    z_axis[:, 1] = 0.0
    z_axis = _normalize(z_axis)

    # Y축 재계산: Z(전방)와 X(좌우)의 외적 -> 완벽한 수직 Y축 생성
    y_axis = np.cross(z_axis, x_axis)
    y_axis = _normalize(y_axis)
    
    # X축도 직교성을 위해 다시 계산
    x_axis = np.cross(y_axis, z_axis)
    x_axis = _normalize(x_axis)

    # 3. 회전 행렬 구성 (T, 3, 3)
    # Global to Local 변환 행렬
    # R = [x_axis, y_axis, z_axis]^T
    R = np.stack([x_axis, y_axis, z_axis], axis=1)

    # 4. 포즈 변환 (Global -> Local)
    # (P_global - Hip_center) * R
    p = seq - hip_center[:, None, :]
    seq_local = np.einsum("tji,tbj->tbi", R, p)

    # 5. Trajectory 생성 (T, 4) -> [RootX, RootY, RootZ, Rotation_Y_Angle]
    # 모델 학습 시 Root의 이동량과 회전량을 알기 위해 필요.
    
    # 회전 각도(Azimuth) 계산: Z축(전방 벡터)을 이용해 atan2로 각도 추출
    # z_axis는 (x, 0, z) 형태이므로 x, z를 이용해 각도 계산
    azimuth = np.arctan2(z_axis[:, 0], z_axis[:, 2]) # 라디안 값
    
    traj = np.zeros((T, 4), dtype=np.float32)
    traj[:, :3] = hip_center # 루트 위치
    traj[:, 3] = azimuth     # 바라보는 방향 (Rotation Y)

    return seq_local.astype(np.float32), traj.astype(np.float32)


def normalize_scale(seq, trajectory):
    processed_seq = seq.copy()
    processed_traj = trajectory.copy()

    # 스케일 기준: 척추 길이 (어깨 중점 ~ 힙 중점)
    shoulders = np.mean(processed_seq[:, [11, 12], :], axis=1)
    hips = np.mean(processed_seq[:, [23, 24], :], axis=1)

    torso_lengths = np.linalg.norm(shoulders - hips, axis=1)
    scale_factor = float(np.mean(torso_lengths))
    
    if scale_factor < 1e-8:
        scale_factor = 1.0

    # 포즈와 궤적(위치) 모두 스케일링
    processed_seq /= scale_factor
    processed_traj[:, :3] /= scale_factor # 위치 정보(xyz)만 나눔, 각도(3번 인덱스)는 그대로 둠
    
    return processed_seq, processed_traj, scale_factor


def create_height_corrected_target(seq):
    processed = seq.copy()
    LFOOT, RFOOT = 31, 32 # 발가락이나 발목 인덱스 사용 (여기선 31, 32가 발끝)
    
    # 모든 프레임, 양발의 Y값 중 최솟값을 찾음 (Global Min)
    all_feet_y = np.concatenate([processed[:, LFOOT, 1], processed[:, RFOOT, 1]])
    
    # 노이즈 방지를 위해 하위 1% 정도를 바닥으로 잡는 것이 안전함
    # min_ground = np.min(all_feet_y) 
    min_ground = np.percentile(all_feet_y, 1)

    processed[:, :, 1] -= min_ground
    
    # 바닥 아래로 내려간 값은 0으로 클램핑 (선택사항)
    # processed[:, :, 1] = np.maximum(processed[:, :, 1], 0)
    
    return processed

def prevent_arm_clipping(seq, threshold=0.15):
    """
    팔이 골반/허벅지를 뚫는 것을 방지 (Clipping Prevention)
    seq: (T, 33, 3) Normalized Pose Data
    threshold: 최소 허용 거리 (이보다 가까우면 밀어냄)
    """
    processed = seq.copy()
    
    # BlazePose 인덱스
    # 23: Left Hip, 24: Right Hip
    # 15: Left Wrist, 16: Right Wrist
    # 13: Left Elbow, 14: Right Elbow
    
    L_HIP, R_HIP = 23, 24
    L_WRIST, R_WRIST = 15, 16
    
    # --- 왼쪽 팔 처리 ---
    # 왼쪽 힙 위치
    l_hip_pos = processed[:, L_HIP, :] 
    # 왼쪽 손목 위치
    l_wrist_pos = processed[:, L_WRIST, :]
    
    # 거리 계산
    dist_l = np.linalg.norm(l_wrist_pos - l_hip_pos, axis=1)
    
    # 충돌 감지 (거리가 threshold보다 작은 프레임 찾기)
    # 단순히 밀어내는 게 아니라, '바깥쪽'으로 밀어야 함
    # 로컬 좌표계에서 X축이 좌우이므로, 왼쪽 팔은 X > 0 방향(또는 <0)으로 밀어야 함
    # (Body Frame 변환 로직에 따라 X축 방향 확인 필요. 보통 왼쪽이 +X or -X)
    
    # 간단한 로직: 현재 손목 위치에서 힙을 뺀 벡터(방향)로 밀어냄
    push_vec_l = l_wrist_pos - l_hip_pos
    push_vec_l = _normalize(push_vec_l) # 단위 벡터
    
    # 침범한 깊이만큼 바깥으로 이동
    mask_l = dist_l < threshold
    # l_hip_pos + (push_vec * threshold) 위치로 강제 이동
    processed[mask_l, L_WRIST, :] = l_hip_pos[mask_l] + push_vec_l[mask_l] * threshold

    # --- 오른쪽 팔 처리 ---
    r_hip_pos = processed[:, R_HIP, :]
    r_wrist_pos = processed[:, R_WRIST, :]
    
    dist_r = np.linalg.norm(r_wrist_pos - r_hip_pos, axis=1)
    
    push_vec_r = r_wrist_pos - r_hip_pos
    push_vec_r = _normalize(push_vec_r)
    
    mask_r = dist_r < threshold
    processed[mask_r, R_WRIST, :] = r_hip_pos[mask_r] + push_vec_r[mask_r] * threshold

    fixed_count = np.sum(mask_l) + np.sum(mask_r)
    total_frames = seq.shape[0]
    percentage = (fixed_count / total_frames) * 100

    return processed, fixed_count


def main():
    print("RUNNING FILE =", __file__)
    paths = glob(f"{RAW_KEYPOINTS_DIR}/*.npz")
    print(f"Found {len(paths)} raw keypoint files")

    # [1] 누적 변수 초기화 (반복문 시작 전!)
    total_all_frames = 0
    total_all_fixed = 0

    for path in paths:
        base = os.path.basename(path).replace(".npz", "")
        print(f"\n[Start Processing] {base} ...")

        try:
            raw_seq = load_raw_keypoints(path)
            if raw_seq.shape[0] == 0:
                print("    ⚠️ ERROR: 데이터 길이가 0입니다.")
                continue

            # Step 0: smoothing
            smooth_seq = smooth_pose_data(raw_seq, window_length=9, polyorder=3)

            # Step 1: body frame transform
            local_seq, traj = process_body_frame_transform(smooth_seq)

            # Step 2: scale normalize
            scaled_seq, scaled_traj, scale_val = normalize_scale(local_seq, traj)

            # Step 2.5: Prevent Arm Clipping (수정 횟수 fix_cnt 받기)
            clipping_fixed_seq, fix_cnt = prevent_arm_clipping(scaled_seq, threshold=0.18)

            # Step 3: height correction
            final_pose = create_height_corrected_target(clipping_fixed_seq)

            # Save
            np.save(os.path.join(OUT_DIR, f"{base}_pose.npy"), final_pose)
            np.save(os.path.join(OUT_DIR, f"{base}_trajectory.npy"), scaled_traj)

            # [2] 현재 영상의 통계 출력
            current_frames = len(final_pose)
            current_rate = (fix_cnt / current_frames) * 100
            print(f"✅ Success: {base}")
            print(f"   - Frames: {current_frames}, Fixed: {fix_cnt} ({current_rate:.2f}%)")

            # [3] 전체 통계에 누적 (저금통에 넣기)
            total_all_frames += current_frames
            total_all_fixed += fix_cnt

        except Exception as e:
            print(f"\n❌ FAIL: {base} 처리 중 오류 발생!")
            print(f"에러 메시지: {e}")
            traceback.print_exc()

    # [4] 반복문이 다 끝나면 종합 결과 출력
    print("\n" + "="*40)
    print("📊 [FINAL DATASET REPORT]")
    print(f"  - Total Videos Processed : {len(paths)}")
    print(f"  - Total Frames Collected : {total_all_frames}")
    print(f"  - Total Clipping Fixed   : {total_all_fixed}")
    
    if total_all_frames > 0:
        avg_rate = (total_all_fixed / total_all_frames) * 100
        print(f"  - Global Correction Rate : {avg_rate:.2f}%")
    print("="*40 + "\n")
    print("All Done.")


if __name__ == "__main__":
    main()