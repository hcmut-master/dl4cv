# DL4CV - Assignment 2: Object Detection on KITTI

## Presentation and website

> [Presentation video](https://drive.google.com/file/d/1C8qNpIMxF2BKQ0BQezIKqeAD0zpHPmBY/view?usp=drive_link)

> [Website](https://hcmut-master.github.io/portfolio/)

## Notebooks

| Notebook | Description | Models |
|----------|-------------|--------|
| `dl4cv-cnn-vs-transformer.ipynb` | Object detection comparing CNN-based vs Transformer-based detectors | YOLOv8s, DETR |
| `dl2cv-a2.ipynb` | Object detection comparing one-stage vs two-stage detectors | YOLOv8s, Faster R-CNN ResNet50-FPN |

## Dataset

- **KITTI Object Detection** — 7,481 images with 39,597 annotated objects
- **Classes (7)**: Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram
- **Split**: 80/20 (5,984 train / 1,497 val)

## Running on Kaggle (Recommended)

Using Kaggle gives you free GPU access (Tesla T4 / P100), which is required for training object detection models.

### Steps

1. Go to [kaggle.com](https://www.kaggle.com/) and sign in (or create an account).
2. Click **"+ Create"** > **"New Notebook"**.
3. Click **"File"** > **"Import Notebook"**, then upload the `.ipynb` file.
4. Enable GPU:
   - Click the **three-dot menu** (top right) or go to **"Settings"** (right sidebar).
   - Under **"Accelerator"**, select **"GPU T4 x2"** or **"GPU P100"**.
5. Click **"Run All"** to execute all cells.

### Notes for Kaggle

- The KITTI dataset is downloaded and preprocessed automatically within the notebook.
- Data format conversion (KITTI -> YOLO -> COCO) is handled in the notebook.
- Training times on GPU:
  - `dl4cv-cnn-vs-transformer.ipynb`:
    - YOLOv8s: ~1 hour
    - DETR: ~6 hours
  - `dl2cv-a2.ipynb`:
    - YOLOv8s: ~1 hour
    - Faster R-CNN ResNet50-FPN: ~2-3 hours

## Running Locally

### Prerequisites

```bash
pip install torch torchvision ultralytics transformers datasets scikit-learn matplotlib seaborn pandas numpy tqdm
```

### Run

```bash
jupyter notebook dl4cv-cnn-vs-transformer.ipynb
# or
jupyter notebook dl2cv-a2.ipynb
```

A CUDA GPU is strongly recommended for training. CPU-only execution will be very slow.
