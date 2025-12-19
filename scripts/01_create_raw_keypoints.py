# scripts/download_and_extract.py

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pytubefix import YouTube
import mediapipe as mp
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_VIDEO_DIR = os.path.join(ROOT, "data", "raw_videos")
RAW_KEYPOINT_DIR = os.path.join(ROOT, "data", "raw_keypoints")
RAW_TEST_DIR = os.path.join(ROOT, "data", "test_keypoints")

os.makedirs(RAW_VIDEO_DIR, exist_ok=True)
os.makedirs(RAW_KEYPOINT_DIR, exist_ok=True)

mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def download_youtube(url):
    yt = YouTube(url)
    name = yt.title.replace(" ", "_").replace("/", "_")
    filepath = f"{RAW_VIDEO_DIR}/{name}.mp4"
    yt.streams.filter(file_extension='mp4').first().download(
        output_path=RAW_VIDEO_DIR,
        filename=f"{name}.mp4"
    )
    print(f"🎬 Downloaded → {filepath}")
    return filepath, name


def extract_3d_keypoints(video_path, dir_path =RAW_KEYPOINT_DIR, name ="Data"):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pose_rows = []

    pbar = tqdm(total=total_frames, desc="Extracting BlazePose 3D",
                ascii=True,          # unicode 막대 → ASCII 막대로 변경
                dynamic_ncols=False  # 윈도우 콘솔 버그 방지
                )
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose_model.process(rgb)

        if res.pose_world_landmarks:
            for i, lm in enumerate(res.pose_world_landmarks.landmark):
                pose_rows.append({
                    "frame": frame_idx,
                    "landmark": i,
                    "x": lm.x,
                    "y": -lm.y,      # flip for unity-like coords
                    "z": lm.z,
                    "visibility": lm.visibility,
                })

        frame_idx += 1
        pbar.update(1)
    cap.release()
    pbar.close()

    df = pd.DataFrame(pose_rows)

    # extract local positions from pelvis


    out_path = f"{dir_path}/{name}.npz"
    np.savez(out_path, data=df.to_numpy())
    print(f"📌 Saved 3D keypoints → {out_path}")

def main():
    # ---------------------------------------------------------
    # [설정] 다운로드할 유튜브 영상 리스트
    # ---------------------------------------------------------
    VIDEO_URLS = [
        "https://youtu.be/ylyV1E_L9rA?si=4I8ER9nV99wOzg4d",  # 영상 1
        "https://youtu.be/example_url_2",                    # 영상 2
        "https://youtu.be/example_url_3",                    # 영상 3
        # ... 계속 추가 가능
    ]
    # ---------------------------------------------------------

    print(f"📋 총 {len(VIDEO_URLS)}개의 영상을 처리합니다.\n")

    for i, url in enumerate(VIDEO_URLS):
        print(f"▶️ [{i+1}/{len(VIDEO_URLS)}] 처리 중: {url}")
        
        try:
            # 1. 유튜브 다운로드
            video_path, name = download_youtube(url)
            
            # 2. 키포인트 추출
            if video_path and os.path.exists(video_path):
                extract_3d_keypoints(video_path, RAW_KEYPOINT_DIR, name)
                print(f"  ✅ 성공: {name}\n")
            else:
                print(f"  ❌ 실패: 다운로드된 파일을 찾을 수 없음 ({url})\n")
                
        except Exception as e:
            print(f"  ❌ 에러 발생 ({url}): {e}\n")
            # 에러가 나도 멈추지 않고 다음 영상으로 넘어갑니다 (continue)
            continue

    print("🎉 모든 작업이 완료되었습니다!")

if __name__ == "__main__":
    main()