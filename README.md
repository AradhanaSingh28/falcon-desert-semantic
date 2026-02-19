# 🚀 Falcon Segmentation
## Multi-Class Semantic Segmentation using Custom U-Net (PyTorch)

---

## 📌 Project Overview

Falcon Segmentation is a deep learning–based semantic segmentation system implemented from scratch using PyTorch.

The model performs **pixel-wise multi-class classification**, assigning a semantic label to every pixel in an image.

This implementation includes:

- Custom Double Convolution blocks
- Encoder–Decoder U-Net architecture
- Skip connections
- Learnable upsampling layers
- Proper mask interpolation handling
- IoU-based evaluation
- Confusion matrix analysis
- Complete modular training pipeline

---

# 🧠 Model Architecture

The model follows a U-Net style encoder–decoder structure.

---

## 🔹 Double Convolution Block

Each block contains:

- 3x3 Convolution
- ReLU
- 3x3 Convolution
- ReLU

Used in both encoder and decoder for strong hierarchical feature extraction.

---

## 🔹 Encoder (Contracting Path)

| Stage | Channels |
|--------|----------|
| Down1  | 3 → 64   |
| Down2  | 64 → 128 |
| Down3  | 128 → 256 |
| Bottleneck | 256 → 512 |

Each stage applies:

- DoubleConv
- MaxPool2D(2)

Purpose:

- Capture low-level and high-level features
- Reduce spatial resolution
- Increase semantic depth

---

## 🔹 Decoder (Expanding Path)

Uses:

- ConvTranspose2D (learnable upsampling)
- Skip connections (concatenation)
- DoubleConv refinement

Purpose:

- Restore spatial resolution
- Improve boundary precision
- Preserve contextual and fine-grained features

---


Raw logits are passed into CrossEntropyLoss.

---

# 📂 Dataset Processing

### Image Processing

- Read using OpenCV
- Convert BGR → RGB
- Resize to 256×256
- Normalize to [0,1]
- Convert to Tensor (C,H,W)

---

### Mask Processing

Critical Step:


Nearest Neighbor interpolation ensures:

- No class corruption
- No blending of labels
- Accurate segmentation masks

Color masks are mapped to integer class IDs before training.

---

# ⚙️ Training Configuration

| Parameter | Value |
|------------|--------|
| Epochs | 20 |
| Batch Size | 2 |
| Learning Rate | 1e-3 |
| Classes | 10 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |

---

# 📊 Evaluation Metrics

## 🔹 1. Intersection over Union (IoU)

IoU is used as the primary evaluation metric.

Formula:


Where:

- Intersection = correctly predicted pixels
- Union = total pixels belonging to predicted or true class

Final IoU is averaged across all classes.

---

## 🔹 2. Confusion Matrix

Below is the confusion matrix generated after validation:

![Segmentation Confusion Matrix](assets/confusion_matrix.png)

The matrix shows:

- True classes (rows)
- Predicted classes (columns)
- Pixel-level classification distribution

Observations:

- Strong diagonal values indicate correct predictions.
- Class 5 shows the highest prediction confidence.
- Some confusion exists between visually similar classes (e.g., class 1 and 4).

---

## 🔹 Model Performance Summary

- Average IoU: **(Insert your final IoU value here, e.g., 0.43 or 0.58)**
- Pixel Accuracy: **(Insert accuracy value if calculated)**
- Strong diagonal dominance in confusion matrix
- Stable convergence over 20 epochs

---

# 📁 Project Structure

Falcon-Segmentation/
│
├── model.py
├── dataset.py
├── train.py
├── utils.py
│
├── assets/
│ └── confusion_matrix.png
│
├── data/
│ ├── train/
│ └── val/
│
└── unet.pth

Training output includes:

- Batch Loss
- Epoch Loss
- Validation IoU
- Execution Time

---

# 💾 Output

- Saved Model: `unet.pth`
- Validation IoU displayed per epoch
- Confusion matrix visualization

---

# 🌟 Key Highlights

- Fully custom U-Net implementation
- Double convolution architecture
- Learnable upsampling
- Proper segmentation mask handling
- IoU-based evaluation
- Confusion matrix performance analysis
- GPU support
- Clean modular code

---

# 🔮 Future Improvements

- Dice Loss integration
- Data augmentation
- Learning rate scheduler
- Mixed precision training
- Early stopping
- Inference pipeline
- Real-time prediction visualization

---

# 🏁 Conclusion

Falcon Segmentation provides a complete semantic segmentation pipeline with:

- Custom architecture design
- Structured preprocessing
- Reliable evaluation metrics
- Quantitative performance validation

This project demonstrates strong understanding of convolutional neural networks and practical segmentation system design.

