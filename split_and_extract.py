import os
import yaml
import cv2
from extract_pose_filtered import process_video_filtered as process_video  # ✅ 用精简版本（只提取身体关键点+6个角度）

VIDEO_DIR = "data"
OUTPUT_JSON_DIR = "data/processed_json"
CONFIG_PATH = "action_segments.yaml"

def cut_video_segment(video_path, start_sec, end_sec, save_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    for _ in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"🎬 Segment saved to {save_path}")

def main():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    video_name = config['video']
    video_path = os.path.join(VIDEO_DIR, video_name)

    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return

    os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

    for segment in config['segments']:
        sec_id = segment['section']
        start = segment['start_time']
        end = segment['end_time']

        # 视频段路径
        segment_video = os.path.join(VIDEO_DIR, f"section_{sec_id}.mp4")
        cut_video_segment(video_path, start, end, segment_video)

        # JSON 输出路径
        segment_json = os.path.join(OUTPUT_JSON_DIR, f"section_{sec_id}.json")
        process_video(segment_video, segment_json)

if __name__ == "__main__":
    main()
