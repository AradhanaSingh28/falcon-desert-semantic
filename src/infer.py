import os
import cv2
import torch
import numpy as np
from torchvision import transforms

from .config import TEST_DIR, IMAGE_SIZE, DEVICE, NUM_CLASSES, BEST_MODEL_PATH
from .model import UNet

def main():
    model = UNet(out_channels=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✅ Loaded:", BEST_MODEL_PATH)

    img_dir = TEST_DIR
    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".png",".jpg",".jpeg"))])

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])

    out_dir = "predictions"
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for f in files[:20]:  # limit for quick demo
            path = os.path.join(img_dir, f)
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

            x = tfm(img).unsqueeze(0).to(DEVICE)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            # save as grayscale class-index mask
            cv2.imwrite(os.path.join(out_dir, f), pred)

    print("✅ Saved predictions to:", out_dir)

if __name__ == "__main__":
    main()