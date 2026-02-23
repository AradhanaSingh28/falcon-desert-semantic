import os
import cv2
import argparse
import numpy as np
import torch
from torchvision import transforms

from src.model import UNet

def list_images(folder):
    exts = (".png",".jpg",".jpeg",".bmp",".tif",".tiff")
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="notebooks/best_model_weighted_dice_by_iou.pth")
    ap.add_argument("--test_root", default="../data/Offroad_Segmentation_testImages")
    ap.add_argument("--out_dir", default="predictions")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--num_classes", type=int, default=6)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    color_dir = os.path.join(args.test_root, "Color_Images")
    os.makedirs(args.out_dir, exist_ok=True)

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std =[0.229,0.224,0.225])
    ])

    model = UNet(in_channels=3, out_channels=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    files = list_images(color_dir)
    print("Found", len(files), "images | device:", device)

    with torch.no_grad():
        for f in files:
            p = os.path.join(color_dir, f)
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)

            x = tfm(img).unsqueeze(0).to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            cv2.imwrite(os.path.join(args.out_dir, f), pred)

    print("✅ Saved predictions to", args.out_dir)

if __name__ == "__main__":
    main()
