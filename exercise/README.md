# DL4CV - Exercise: Image Classification on CIFAR-10

## Notebooks

| Notebook | Description | Models |
|----------|-------------|--------|
| `dl4cv.ipynb` | Image classification with CNN and Transformer architectures | Softmax Regression, MLP, CNN (CustomResNet), ViT, Custom ViT, Hybrid CNN-ViT, Overlapping ViT |
| `dl4cv-lstm-gru.ipynb` | Image classification with sequence models (row-wise representation) | LSTM, GRU |

## Running on Kaggle (Recommended)

Using Kaggle gives you free GPU access (Tesla T4 / P100), which significantly speeds up training.

### Steps

1. Go to [kaggle.com](https://www.kaggle.com/) and sign in (or create an account).
2. Click **"+ Create"** > **"New Notebook"**.
3. Click **"File"** > **"Import Notebook"**, then upload the `.ipynb` file.
4. Enable GPU:
   - Click the **three-dot menu** (top right) or go to **"Settings"** (right sidebar).
   - Under **"Accelerator"**, select **"GPU T4 x2"** or **"GPU P100"**.
5. Click **"Run All"** to execute all cells.

### Notes for Kaggle

- CIFAR-10 is downloaded automatically via `torchvision.datasets.CIFAR10` — no manual dataset setup needed.
- `dl4cv.ipynb` uses `torch.device("mps")` for Mac. On Kaggle, the notebook auto-detects CUDA, so this is handled gracefully.
- Training times on GPU:
  - `dl4cv.ipynb`: ~15-30 min (depending on model)
  - `dl4cv-lstm-gru.ipynb`: ~2 min (15 epochs)

## Running Locally

### Prerequisites

```bash
pip install torch torchvision torchinfo scikit-learn matplotlib seaborn pandas numpy tqdm
```

### Run

```bash
jupyter notebook dl4cv.ipynb
# or
jupyter notebook dl4cv-lstm-gru.ipynb
```

If you have a CUDA GPU, PyTorch will use it automatically. On Mac, MPS backend is used as fallback.
