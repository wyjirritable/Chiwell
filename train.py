import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import BaduanjinDataset
from model import LSTMClassifier
import os

# 📌 超参数设置
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
MAX_FRAMES = 60
NUM_CLASSES = 10

# 📂 路径设置
JSON_DIR = "./data/processed_json"
SAVE_PATH = "./output/model.pt"
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# ✅ 准备数据集
dataset = BaduanjinDataset(json_dir=JSON_DIR, max_frames=MAX_FRAMES)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ✅ 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(input_dim=6, num_classes=NUM_CLASSES).to(device)

# ✅ 损失函数 & 优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ✅ 训练循环
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X, y in train_loader:
        X = X.to(device)           # [B, T, 6]
        y = y.to(device)           # [B]

        optimizer.zero_grad()
        outputs = model(X)         # [B, num_classes]
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f} - Acc: {acc:.4f}")

# ✅ 保存模型
torch.save(model.state_dict(), SAVE_PATH)
print(f"✅ 模型已保存至: {SAVE_PATH}")
