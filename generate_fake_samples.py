import os
import json
import numpy as np
import random

# 原始样本路径（可改为你已有的）
SOURCE_FILE = "data/processed_json/section_0.json"
OUTPUT_DIR = "data/processed_json"
N_SAMPLES_PER_CLASS = 5   # 每个 section 复制几个假样本
NOISE_STD = 3.0           # 每个角度扰动的标准差（单位：度）

def generate_augmented_frames(frames, noise_std):
    new_frames = []
    for frame in frames:
        new_frame = frame.copy()
        new_angles = {}
        for k, v in frame.get("angles", {}).items():
            if v is not None:
                noisy_v = v + np.random.normal(0, noise_std)
                new_angles[k] = float(np.clip(noisy_v, 0, 180))  # 限制角度范围
            else:
                new_angles[k] = None
        new_frame["angles"] = new_angles
        new_frames.append(new_frame)
    return new_frames

def main():
    for section_id in range(10):  # 0~9 每段生成
        source_file = f"{OUTPUT_DIR}/section_{section_id}.json"
        if not os.path.exists(source_file):
            print(f"⚠️ 缺失原始文件: {source_file}")
            continue

        with open(source_file, 'r', encoding='utf-8') as f:
            original_frames = json.load(f)

        for i in range(N_SAMPLES_PER_CLASS):
            fake_frames = generate_augmented_frames(original_frames, noise_std=NOISE_STD)
            fake_path = f"{OUTPUT_DIR}/section_{section_id}_aug{i+1}.json"
            with open(fake_path, 'w', encoding='utf-8') as f:
                json.dump(fake_frames, f, indent=2, ensure_ascii=False)
            print(f"✅ 生成伪样本: {fake_path}")

if __name__ == "__main__":
    main()
