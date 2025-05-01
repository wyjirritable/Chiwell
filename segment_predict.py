import torch
import torch.nn.functional as F
import json
import numpy as np
from model import LSTMClassifier

# 📌 参数设置
MODEL_PATH = "./output/model.pt"
INPUT_JSON = "data/processed_json/full_video.json"
NUM_CLASSES = 10
ANGLE_KEYS = [
    "left_elbow_angle", "right_elbow_angle",
    "left_knee_angle", "right_knee_angle",
    "left_hip_angle", "right_hip_angle"
]
FPS = 30                     # 视频帧率
WINDOW_SECONDS = 30          # 每段识别时长（秒）
WINDOW_SIZE = FPS * WINDOW_SECONDS  # 每段识别窗口大小（帧）
STRIDE = WINDOW_SIZE                # 每30秒预测一次

# ✅ ID 映射表
ID_TO_LABEL = {
    0: "两手托天理三焦",
    1: "左右开弓似射雕",
    2: "调理脾胃须单举",
    3: "五劳七伤往后瞧",
    4: "摇头摆尾去心火",
    5: "两手攀足固肾腰",
    6: "攒拳怒目增气力",
    7: "背后七颠百病消",
    8: "预备式",
    9: "收势"
}

def load_json_sequence(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        frames = json.load(f)

    angle_seq = []
    for frame in frames:
        angles = frame.get("angles", {})
        vec = [float(angles.get(k, 0.0) or 0.0) for k in ANGLE_KEYS]
        angle_seq.append(vec)

    return np.array(angle_seq, dtype=np.float32)  # shape: [T, 6]

def predict_segment(model, segment_tensor):
    model.eval()
    with torch.no_grad():
        logits = model(segment_tensor.unsqueeze(0))  # [1, T, 6]
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        return pred, probs.squeeze().tolist()

def smooth_segments(preds):
    segments = []
    last_label = None
    start_idx = 0
    for i, label in enumerate(preds):
        if label != last_label:
            if last_label is not None:
                segments.append((start_idx, i, last_label))
            start_idx = i
            last_label = label
    segments.append((start_idx, len(preds), last_label))
    return segments

def main():
    # ✅ 加载模型
    model = LSTMClassifier(input_dim=6, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))

    # ✅ 加载角度序列
    angle_seq = load_json_sequence(INPUT_JSON)
    T = angle_seq.shape[0]

    preds = []
    for start in range(0, T - WINDOW_SIZE + 1, STRIDE):
        window = angle_seq[start: start + WINDOW_SIZE]
        window_tensor = torch.tensor(window, dtype=torch.float32)
        pred, prob = predict_segment(model, window_tensor)
        preds.append(pred)

    # ✅ 合并连续相同片段
    segments = smooth_segments(preds)

    # ✅ 打印最终结果
    print("\n📊 动作段识别结果（滑动窗口识别）：\n")
    for seg_start, seg_end, label_id in segments:
        frame_start = seg_start * STRIDE
        frame_end = (seg_end * STRIDE) + WINDOW_SIZE
        time_start = frame_start / FPS
        time_end = frame_end / FPS
        label_name = ID_TO_LABEL.get(label_id, f"动作 {label_id}")
        print(f"[{time_start:.1f}s ~ {time_end:.1f}s] → {label_name}")

if __name__ == "__main__":
    main()
