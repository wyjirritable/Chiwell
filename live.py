import cv2
import mediapipe as mp
import time
import json
import numpy as np
import os

# 路径兼容性修复
with open(os.path.join("video1", "standard_pose_data.json"), "r") as f:
    standard_data = json.load(f)

def find_closest_standard_frame(timestamp, standard_data):
    closest_frame = None
    min_diff = float('inf')
    for frame in standard_data:
        if "angles" not in frame:
            continue
        diff = abs(timestamp - frame['timestamp'])
        if diff < min_diff:
            min_diff = diff
            closest_frame = frame
    return closest_frame

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 只保留身体主要连接线（去掉面部连接）
BODY_CONNECTIONS = [
    (11,13), (13,15), (12,14), (14,16),   # 手臂
    (11,12),                              # 肩膀连线
    (23,24),                              # 髋连线
    (11,23), (12,24),                     # 躯干
    (23,25), (25,27), (24,26), (26,28),   # 腿部
]

cap = cv2.VideoCapture(0)
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time() - start_time
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    image_output = frame.copy()

    cv2.putText(image_output, f"Standard Time: {current_time:.2f}s", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        get_xy = lambda i: [lm[i].x, lm[i].y]

        try:
            angles = {
                "left_elbow_angle": calculate_angle(get_xy(11), get_xy(13), get_xy(15)),
                "right_elbow_angle": calculate_angle(get_xy(12), get_xy(14), get_xy(16)),
                "left_knee_angle": calculate_angle(get_xy(23), get_xy(25), get_xy(27)),
                "right_knee_angle": calculate_angle(get_xy(24), get_xy(26), get_xy(28)),
                "left_hip_angle": calculate_angle(get_xy(11), get_xy(23), get_xy(25)),
                "right_hip_angle": calculate_angle(get_xy(12), get_xy(24), get_xy(26))
            }

            # 🔸 对称性差值
            symmetry_diff = (
                abs(angles["left_elbow_angle"] - angles["right_elbow_angle"]) +
                abs(angles["left_knee_angle"] - angles["right_knee_angle"]) +
                abs(angles["left_hip_angle"] - angles["right_hip_angle"])
            ) / 3

        except:
            angles = {}
            symmetry_diff = 999

        standard_frame = find_closest_standard_frame(current_time, standard_data)
        feedback = "Tracking..."

        if standard_frame and angles:
            std_angles = standard_frame["angles"]
            diffs = []
            for key in angles:
                if key in std_angles:
                    diffs.append(abs(angles[key] - std_angles[key]))

            avg_diff = sum(diffs) / len(diffs) if diffs else 999

            if avg_diff < 15 and symmetry_diff < 15:
                feedback = "Good posture!"
            elif symmetry_diff >= 15:
                feedback = "Adjust left/right balance"
            else:
                feedback = "Please adjust your pose"

            # 展示角度差与对称性
            cv2.putText(image_output, f"Angle diff: {avg_diff:.1f}", (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
            cv2.putText(image_output, f"Symmetry diff: {symmetry_diff:.1f}", (30, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 255), 2)

        cv2.putText(image_output, feedback, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # 🔸 自定义只画身体骨架
        # 手动绘制关键点和连接线（跳过面部）
        landmarks = results.pose_landmarks.landmark
        image_h, image_w = image_output.shape[:2]
        allowed_indices = set(range(11, 29))  # 只绘制身体关键点

        # 绘制连接线
        for start_idx, end_idx in BODY_CONNECTIONS:
            if start_idx in allowed_indices and end_idx in allowed_indices:
                x1, y1 = int(landmarks[start_idx].x * image_w), int(landmarks[start_idx].y * image_h)
                x2, y2 = int(landmarks[end_idx].x * image_w), int(landmarks[end_idx].y * image_h)
                cv2.line(image_output, (x1, y1), (x2, y2), (0, 100, 255), 2)

        # 绘制关键点
        for idx in allowed_indices:
            x, y = int(landmarks[idx].x * image_w), int(landmarks[idx].y * image_h)
            cv2.circle(image_output, (x, y), 4, (0, 255, 0), -1)

    cv2.imshow("Angle-Based Real-Time Pose Feedback", image_output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
