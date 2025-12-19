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
        # "https://youtu.be/ylyV1E_L9rA?si=4I8ER9nV99wOzg4d",  # 영상 1
        # "https://youtube.com/shorts/KC1AW3Y3DMQ?si=NiMYEmJzUX8RSDF7",
        # "https://youtube.com/shorts/W9PxV2VIT2w?si=oG5CV2GnhO5LEDoR",
        # "https://youtube.com/shorts/zEA1FoiHCBE?si=DyT_EN4A1TB-cB_p",
        # "https://youtube.com/shorts/PVm1TRJ9kRM?si=q2BckrEa_r2bo847",
        # "https://youtube.com/shorts/ZsyJeQMYDuY?si=9zsYMg8nPVIdGlxU",
        # "https://youtube.com/shorts/nL4brQI1J6A?si=aKmenu_RUd7C1KGp",
        # "https://youtube.com/shorts/W1sYGYtSFR0?si=c8T3rx0We8U_is6c",
        # "https://youtube.com/shorts/8tHES29GmUc?si=c4TvHD529BuutLES",
        # "https://youtube.com/shorts/UFLyEwwj_T8?si=BTcjvZwQDAwlNoej",

        # "https://youtube.com/shorts/O5mKEHndUZk?si=JXEGpvKg6J_E4msy",
        # "https://youtube.com/shorts/NXTchWX7XFc?si=xai6r1EHRV5KjyOH",
        # "https://youtube.com/shorts/1AKiz5Ahu30?si=hDcerpOAjQqDGpUc",
        # "https://youtube.com/shorts/AQnIPrGsc9c?si=Jpckln-Wb6vdJtt3",
        # "https://youtube.com/shorts/vqN3cXmhejs?si=-F7jAMU-4nC4Gt1_",
        # "https://youtube.com/shorts/jxkqG7yyUrs?si=yJjajDNsuPIZTuum",
        # "https://youtube.com/shorts/542W2lCfoAw?si=SjnOu55r9JrV9puy",
        # "https://youtube.com/shorts/5gHRVgqN8A0?si=5hdEC20H5wF4UTW1",
        # "https://youtube.com/shorts/jl73pznX9Uc?si=t7vbhO1g4ScYjngs",
        # "https://youtube.com/shorts/hc5V41cTIz8?si=Pt8kkLzH_F1G-VjH",

        # "https://youtube.com/shorts/sxh4ZjkNcFg?si=4ukwmE5Kz3tvTzl9",
        # "https://youtube.com/shorts/UtGyammnPZo?si=jblcMay4147q9sqm",
        # "https://youtube.com/shorts/HdMDqtxVuts?si=ztrB469J06ZtWqQr",
        # "https://youtube.com/shorts/3e69wyEKUqQ?si=QK-opOPOO-V4RnhB",
        # "https://youtube.com/shorts/Sijq3ScdM2c?si=2l_jY_IdBfzQzgvS",
        # "https://youtube.com/shorts/XIKplfAQ0W8?si=tcPsb-_IUxWCvgJo",
        # "https://youtube.com/shorts/8Y0NEUsRxrI?si=sYahcji64lbHy29P",
        # "https://youtube.com/shorts/TvhSEPrnYUc?si=hCe0NezcHT4gOOCF",
        # "https://youtube.com/shorts/AfNB1DNXOnw?si=53uWOjjaw7MvmFPb",
        # "https://youtube.com/shorts/evOU7PVkqG4?si=a0eIObtulPkjI936",
        # "https://youtube.com/shorts/e25buagBqso?si=BbYfLZUELNHHUf4x",

        # "https://youtube.com/shorts/-2oQ0-ykQ3A?si=9TiRv6-OJ1Mt_voz",
        # "https://youtube.com/shorts/YMhtRttrtnU?si=aVuW0Q9ZrwFoeEl6",
        # "https://youtube.com/shorts/H52Z928zVIM?si=k_AySpbl8ubrU-27",
        # "https://youtube.com/shorts/qt5k2aPP4Ds?si=dwlALAsFcgbprpZX",
        # "https://youtube.com/shorts/7XuLgEjD9PU?si=EL7_4ulD0xiRVf4u",
        # "https://youtube.com/shorts/3Pv_WKp1dAk?si=q1BqdDe6qTWgVcSd",
        # "https://youtube.com/shorts/AGQNC8MTAvc?si=Dkl7JbIK-YPzZx8L",
        # "https://youtube.com/shorts/X64eBHBIUJo?si=id-xZkazBqIKgdvC",
        # "https://youtube.com/shorts/6m73fWP8hTs?si=_3mozwCR2pEvcz3m",
        # "https://youtube.com/shorts/LEpuqlFkq64?si=HXmLUTaBsPFziADX",

        # #Sport#
        # "https://youtube.com/shorts/D0_atBgcZ_0?si=YXJHctyS2r0Ri2TL",
        # "https://youtube.com/shorts/4LyPqANVoDE?si=FJGTqSlDPQnqVcie",
        # "https://youtube.com/shorts/q0nt8CyyteE?si=ODRsyQRj3U9WgAre",
        # "https://youtube.com/shorts/1YI2HvMsKug?si=vcVQs3Hd9R6lQxb0",
        # "https://youtube.com/shorts/qdBLvdu5Y8k?si=Ce1GUO8ny7ZTkvGu",
        # "https://youtube.com/shorts/5Vj0BzGYBok?si=rgn-e0GtI5oie1w5",
        # "https://youtube.com/shorts/TnOkq6KfHsM?si=uIayNAg3r-PK8C0f",
        # "https://youtube.com/shorts/Gt9hlRMXDXc?si=y-99M4j-Mz2mTw2j",
        # "https://youtube.com/shorts/NEhrf-RDg4o?si=Kvvx5-AGODGHafRs",
        # "https://youtube.com/shorts/CSZVGGseOZ4?si=CzhbzO26gAb_zPTn"

        "https://youtube.com/shorts/8tHES29GmUc?si=6B3m3vn_6GjpLNcX"


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