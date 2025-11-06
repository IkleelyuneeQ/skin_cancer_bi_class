# Skin Cancer Classification — ResNet18 (Binary: Cancer vs. Non_Cancer)

> Transfer learning pipeline for dermatoscopic image classification (Cancer / Non_Cancer)
> built with **PyTorch + torchvision**. Includes normalization from data,
> frozen ResNet18 backbone, custom classifier head, ReduceLROnPlateau,
> early stopping & checkpointing — with clean evaluation and visualizations.

<p align="left">
  <img src="https://img.shields.io/badge/PyTorch-2.x-red" />
  <img src="https://img.shields.io/badge/torchvision-0.x-orange" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" />
</p>

## Problem Statement
Early detection of skin cancer is critical. This project trains a model to classify dermatoscopic images as **Cancer** or **Non_Cancer**, prioritizing **high recall for Cancer** (minimize missed cancers) while keeping overall accuracy competitive.

---

## Dataset
- Two classes: `Cancer`, `Non_Cancer`
- Layout:
  ```text
  data/
    Skin_Data/
      train/        # 84 images total (balanced)
      test/         # 204 images total
  ```
- A validation set is created via **80/20 split** on `train/` with a fixed seed.

### Normalization (computed from training set)
```
mean = [0.6265, 0.4395, 0.3652]
std  = [0.1900, 0.1746, 0.1637]
```

---

## Model
- Base: **ResNet18** (ImageNet weights), **backbone frozen**
- New classification head:
  ```python
  nn.Sequential(
      nn.Linear(512, 256),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(256, 2)
  )
  ```
- Loss: `CrossEntropyLoss`
- Optimizer: `Adam(lr=1e-3)`
- Scheduler: `ReduceLROnPlateau`
- Early Stopping: Patience = 10 epochs
- Checkpointing: best by validation loss (`best_model.pth`)

---

Core transforms:
```python
transform_norm = transforms.Compose([
    ConvertToRGB(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])
```

---

## Results

### Validation (16 images)
| Class       | Precision | Recall | F1-score |
|-------------|-----------|--------|----------|
| Cancer      | 0.8182    | **0.9000** | 0.8571 |
| Non_Cancer  | 0.8000    | 0.6667 | 0.7273 |

**Validation Accuracy:** 81.25%

### Test (204 images)
| Class       | Precision | Recall | F1-score |
|-------------|-----------|--------|----------|
| Cancer      | 0.4211    | **0.9524** | 0.5839 |
| Non_Cancer  | **0.9817**| 0.6605 | 0.7897 |

**Overall Test Accuracy:** 72.06%  
**Observation:** The model is **safety-biased** — it strongly prioritizes **Cancer recall (95%)**, trading off with more **false positives** (acceptable for clinical triage).

---

## Figures
- Validation confusion matrix
- Test confusion matrix
- Training/validation loss & accuracy
- Learning rate schedule

> See notebook

---

## Engineering Notes
- Reproducibility: fixed seeds for splits and dataloaders
- Frozen backbone ensures stability on small data
- `ReduceLROnPlateau` + EarlyStopping mitigate overfitting
- Clean separation of `train/evaluate/predict` functions

# skin_cancer_bi_class
