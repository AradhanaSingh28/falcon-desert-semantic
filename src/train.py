import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import (
    TRAIN_DIR, VAL_DIR, IMAGE_SIZE, BATCH_SIZE, EPOCHS, LR,
    DEVICE, NUM_CLASSES, MODEL_DIR, BEST_MODEL_PATH
)
from .dataset import DesertDataset
from .model import UNet
from .metrics import mean_iou

def compute_class_weights(loader, num_classes, device):
    counts = torch.zeros(num_classes, device=device)
    for _, masks in loader:
        masks = masks.to(device)
        for c in range(num_classes):
            counts[c] += (masks == c).sum()
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return weights

def dice_loss(logits, targets, num_classes, eps=1e-6):
    probs = F.softmax(logits, dim=1)
    targets_1h = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    inter = (probs * targets_1h).sum(dim=(0, 2, 3))
    union = (probs + targets_1h).sum(dim=(0, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_ds = DesertDataset(TRAIN_DIR, image_size=IMAGE_SIZE, normalize=True)
    val_ds   = DesertDataset(VAL_DIR,   image_size=IMAGE_SIZE, normalize=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = UNet(out_channels=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # weights + CE
    weights = compute_class_weights(train_loader, NUM_CLASSES, DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_miou = -1.0

    print("Device:", DEVICE)
    print("Train samples:", len(train_ds), "Val samples:", len(val_ds))
    print("Num classes:", NUM_CLASSES)
    print("Saving best model to:", BEST_MODEL_PATH)

    for epoch in range(1, EPOCHS + 1):
        start = time.time()

        # ---- Train ----
        model.train()
        train_loss = 0.0

        for imgs, masks in train_loader:
            imgs  = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(imgs)
            loss = criterion(logits, masks) + 0.5 * dice_loss(logits, masks, NUM_CLASSES)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))

        # ---- Val ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs  = imgs.to(DEVICE)
                masks = masks.to(DEVICE)

                logits = model(imgs)
                loss = criterion(logits, masks) + 0.5 * dice_loss(logits, masks, NUM_CLASSES)
                val_loss += loss.item()

        val_loss /= max(1, len(val_loader))

        per_class, miou = mean_iou(model, val_loader, NUM_CLASSES, DEVICE)
        epoch_time = round(time.time() - start, 1)

        print(f"\nEpoch {epoch}/{EPOCHS} | time={epoch_time}s")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | mIoU: {miou:.4f}")
        print("Per-class IoU:", per_class.numpy())

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print("✅ Saved best model. Best mIoU:", round(best_miou, 4))

    print("\nDone. Best mIoU:", round(best_miou, 4))

if __name__ == "__main__":
    main()