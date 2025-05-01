import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from glob import glob

class BaduanjinDataset(Dataset):
    def __init__(self, json_dir, max_frames=60, angle_keys=None):
        self.json_paths = sorted(glob(os.path.join(json_dir, "section_*.json")))
        self.max_frames = max_frames
        self.angle_keys = angle_keys or [
            "left_elbow_angle",
            "right_elbow_angle",
            "left_knee_angle",
            "right_knee_angle",
            "left_hip_angle",
            "right_hip_angle",
        ]
        self.samples = self._load_all_samples()

    def _load_all_samples(self):
        data = []
        for path in self.json_paths:
            with open(path, 'r', encoding='utf-8') as f:
                frames = json.load(f)

            angle_seq = []
            for frame in frames:
                angles = frame.get("angles", {})
                angle_vec = [angles.get(k, 0.0) for k in self.angle_keys]
                angle_seq.append(angle_vec)

            angle_seq = self._pad_or_truncate(np.array(angle_seq))  # [T, 6]

            # extract label from filename, e.g., section_3.json → label=3
            label = int(os.path.basename(path).split("_")[1].split(".")[0])
            data.append((angle_seq, label))
        return data

    def _pad_or_truncate(self, sequence):
        T, D = sequence.shape
        if T == self.max_frames:
            return sequence
        elif T < self.max_frames:
            pad_len = self.max_frames - T
            pad = np.zeros((pad_len, D))
            return np.vstack([sequence, pad])
        else:  # T > max_frames → sample均匀选帧
            idx = np.linspace(0, T - 1, self.max_frames).astype(int)
            return sequence[idx]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y = self.samples[idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
