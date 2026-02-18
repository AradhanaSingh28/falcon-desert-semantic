import os
import cv2
import torch
from torch.utils.data import Dataset

class FalconDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=256):
        """
        image_dir : path to RGB images
        mask_dir  : path to segmentation masks
        img_size  : resize dimension (default 256x256)
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size

        # list all image filenames
        self.image_names = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]

        # construct full paths
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)

        # =========================
        # LOAD AND PROCESS IMAGE
        # =========================
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # resize image
        image = cv2.resize(image, (self.img_size, self.img_size))

        # convert to tensor and normalize
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # =========================
        # LOAD AND PROCESS MASK
        # =========================
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # resize mask (IMPORTANT: nearest interpolation)
        mask = cv2.resize(mask, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_NEAREST)

        # convert to tensor
        mask = torch.from_numpy(mask)

        # reshape to (H*W, 3)
        colors = mask.view(-1, 3)

        # find unique colors and map to class IDs
        unique_colors, inverse_indices = torch.unique(
            colors, dim=0, return_inverse=True
        )

        label_mask = inverse_indices.view(self.img_size, self.img_size)

        mask = label_mask.long()

        return image, mask