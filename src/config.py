import os
import torch

# -------- Paths --------
DATASET_ROOT = os.path.join("..", "data")

TRAIN_DIR = os.path.join(DATASET_ROOT, "Offroad_Segmentation_Training_Dataset", "train")
VAL_DIR   = os.path.join(DATASET_ROOT, "Offroad_Segmentation_Training_Dataset", "val")

# Test images are ONLY for inference (never training)
TEST_DIR  = os.path.join(DATASET_ROOT, "Offroad_Segmentation_testImages")

# -------- Training --------
IMAGE_SIZE  = 128
BATCH_SIZE  = 4
EPOCHS      = 40
LR          = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------- Labels --------
# Raw label values found in masks (update if your scan finds more)
LABEL_VALUES = [0, 1, 2, 3, 27, 39]
LABEL_MAP = {v: i for i, v in enumerate(LABEL_VALUES)}
NUM_CLASSES = len(LABEL_VALUES)

# -------- Output --------
MODEL_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model_weighted_dice_128_full.pth")