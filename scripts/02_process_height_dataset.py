import os
import numpy as np
import pandas as pd
from glob import glob
import sys
import traceback
import time

# (권장) 스레드 꼬임 방지
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
    [수정됨] 포즈를 로컬 좌표계로 변환하고, 
    Trajectory(루트 위치 + 회전) 정보를 함께 반환합니다.
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

    # 🔥 Roll 제거: Z축을 수평면에 투영 (y성분 0으로 만듦)
    z_axis[:, 1] = 0.0
    z_axis = _normalize(z_axis)

    # Y축 재계산: Z(전방)와 X(좌우)의 외적 -> 완벽한 수직 Y축 생성
    y_axis = np.cross(z_axis, x_axis)
    y_axis = _normalize(y_axis)
    
    # X축도 직교성을 위해 다시 계산 (선택사항이나 안전함)
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
    # 모델 학습 시 Root의 이동량과 회전량을 알기 위해 필요합니다.
    
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


def main():
    print("RUNNING FILE =", __file__)
    paths = glob(f"{RAW_KEYPOINTS_DIR}/*.npz")
    print(f"Found {len(paths)} raw keypoint files")

    for path in paths:
        base = os.path.basename(path).replace(".npz", "")
        print(f"\n[Start Processing] {base} ...")

        try:
            # print("  > Loading .npz file...")
            raw_seq = load_raw_keypoints(path)
            # print(f"    - Loaded Sequence Shape: {raw_seq.shape} dtype={raw_seq.dtype}")

            if raw_seq.shape[0] == 0:
                print("    ⚠️ ERROR: 데이터 길이가 0입니다.")
                continue

            # Step 0: smoothing
            # print("  > Step 0: Smoothing...")
            smooth_seq = smooth_pose_data(raw_seq, window_length=9, polyorder=3)

            # Step 1: body frame transform (리턴값 2개 받도록 수정됨)
            # print("  > Step 1: Body-Frame Local Transform...")
            local_seq, traj = process_body_frame_transform(smooth_seq)

            # Step 2: scale normalize
            # print("  > Step 2: Scale Normalization...")
            scaled_seq, scaled_traj, scale_val = normalize_scale(local_seq, traj)

            # Step 3: height correction (Target Data 생성)
            # print("  > Step 3: Height Correction...")
            final_pose = create_height_corrected_target(scaled_seq)

            # Save
            # print("  > Saving...")
            np.save(os.path.join(OUT_DIR, f"{base}_pose.npy"), final_pose)
            np.save(os.path.join(OUT_DIR, f"{base}_trajectory.npy"), scaled_traj)

            print(f"✅ Success: {base} (Frames: {len(final_pose)}, Scale: {scale_val:.4f})")

        except Exception as e:
            print(f"\n❌ FAIL: {base} 처리 중 오류 발생!")
            print(f"에러 메시지: {e}")
            traceback.print_exc()

    print("\nAll Done.")


if __name__ == "__main__":
    main()