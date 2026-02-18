import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from model import UNet
from dataset import FalconDataset
from utils import compute_iou

# -----------------------------
# Settings
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 10
BATCH_SIZE = 2
IMG_SIZE = 256
MAX_BATCHES = None  # Set to an integer to test only a few batches

# -----------------------------
# Load model
# -----------------------------
model = UNet(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(
    "C:/Users/Pankhudi Gupta/Downloads/falcon-desert-semantic/unet.pth",
    map_location=DEVICE
))
model.eval()
print("✅ Model loaded!")

# -----------------------------
# Validation dataset
# -----------------------------
val_dataset = FalconDataset(
    image_dir="C:/Users/Pankhudi Gupta/Downloads/falcon-desert-semantic/data/val/Color_Images",
    mask_dir="C:/Users/Pankhudi Gupta/Downloads/falcon-desert-semantic/data/val/Segmentation",
    img_size=IMG_SIZE
)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)  # change num_workers>0 on Linux/Mac

# -----------------------------
# Evaluation
# -----------------------------
all_preds = []
all_masks = []
ious_per_image = []

with torch.no_grad():
    for batch_idx, (imgs, masks) in enumerate(val_loader):
        if MAX_BATCHES and batch_idx >= MAX_BATCHES:
            break

        print(f"Processing batch {batch_idx+1}...")
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE).long()

        # Forward pass
        preds = model(imgs)
        pred_labels = torch.argmax(preds, dim=1)

        # IoU per batch
        iou = compute_iou(preds, masks, NUM_CLASSES)
        ious_per_image.append(iou)

        all_preds.append(pred_labels.cpu())
        all_masks.append(masks.cpu())
        print(f"Batch {batch_idx+1} done | IoU: {iou:.4f}")

# Flatten all predictions and masks for confusion matrix
all_preds_flat = torch.cat(all_preds).view(-1).numpy()
all_masks_flat = torch.cat(all_masks).view(-1).numpy()

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(all_masks_flat, all_preds_flat, labels=list(range(NUM_CLASSES)))

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Segmentation Confusion Matrix")
plt.show()

# -----------------------------
# IoU Graph
# -----------------------------
mean_iou = np.mean(ious_per_image)
print(f"\n📊 Mean IoU over validation set: {mean_iou:.4f}")

plt.figure(figsize=(8,5))
plt.plot(ious_per_image, marker='o')
plt.title("IoU per Validation Batch")
plt.xlabel("Batch index")
plt.ylabel("IoU")
plt.grid(True)
plt.show()
