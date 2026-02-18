import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time

print("🚀 Starting training script...")

from dataset import FalconDataset
from model import UNet
from utils import compute_iou

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

NUM_CLASSES = 10
BATCH_SIZE = 2   # smaller for stability
EPOCHS = 20
LR = 1e-3

print("📂 Loading dataset...")

train_ds = FalconDataset(
    "data/train/Color_Images",
    "data/train/Segmentation"
)

val_ds = FalconDataset(
    "data/val/Color_Images",
    "data/val/Segmentation"
)

print("Train samples:", len(train_ds))
print("Val samples:", len(val_ds))

print("📦 Creating DataLoaders...")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)

print("🧠 Building model...")

model = UNet(NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

print("🔥 Starting training loop")

for epoch in range(EPOCHS):
    start_time = time.time()

    model.train()
    train_loss = 0

    for batch_idx, (imgs, masks) in enumerate(train_loader):
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE).long()

        preds = model(imgs)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % 5 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss {loss.item():.4f}")

    avg_loss = train_loss / len(train_loader)
    print(f"\n✅ Epoch {epoch+1} Training Loss: {avg_loss:.4f}")

    # ===== Validation =====
    model.eval()
    iou_total = 0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE).long()

            preds = model(imgs)
            iou_total += compute_iou(preds, masks, NUM_CLASSES)

    avg_iou = iou_total / len(val_loader)
    print(f"📊 Validation IoU: {avg_iou:.4f}")

    print(f"⏱ Epoch time: {time.time() - start_time:.2f}s\n")

print("💾 Saving model...")
torch.save(model.state_dict(), "unet.pth")

print("🎉 Training complete!")