import torch
from torch.utils.data import DataLoader

from .config import VAL_DIR, IMAGE_SIZE, BATCH_SIZE, DEVICE, NUM_CLASSES, BEST_MODEL_PATH
from .dataset import DesertDataset
from .model import UNet
from .metrics import mean_iou

def main():
    val_ds = DesertDataset(VAL_DIR, image_size=IMAGE_SIZE, normalize=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = UNet(out_channels=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    print("✅ Loaded:", BEST_MODEL_PATH)

    per_class, miou = mean_iou(model, val_loader, NUM_CLASSES, DEVICE)
    print("Per-class IoU:", per_class.numpy())
    print("Mean IoU:", miou)

if __name__ == "__main__":
    main()