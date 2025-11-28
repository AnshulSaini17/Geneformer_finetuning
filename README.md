# Geneformer Fine-tuning for Cell Type Classification

Fine-tuning Geneformer models for cardiomyocyte subtype classification using single-cell RNA-seq data.

## Overview

This project provides a clean, modular implementation for fine-tuning Geneformer on cell classification tasks. The code supports both local training and Google Colab with GPU acceleration.

## Setup

### Local Installation

```bash
# Clone repository
git clone https://github.com/yourusername/geneformer-finetuning.git
cd geneformer-finetuning

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Option 1: Use setup script (easiest)
bash setup.sh

# Option 2: Manual installation
pip install -r requirements.txt
pip install --no-deps git+https://huggingface.co/ctheodoris/Geneformer
```

### Google Colab

```python
# Clone repo and install
!git clone https://github.com/yourusername/geneformer-finetuning.git
%cd geneformer-finetuning
!bash setup.sh

# Mount Google Drive (if your dataset is there)
from google.colab import drive
drive.mount('/content/drive')
```

## Usage

### 1. Prepare Your Dataset

Your dataset should be an Arrow file with:
- `input_ids`: Tokenized gene expression data
- `cell_type`: Cell type labels
- Other metadata columns

### 2. Configure Training

Edit `configs/config.yaml`:

```yaml
data:
  dataset_file: "dataset.arrow"
  cell_types:
    - "Cardiomyocyte1"
    - "Cardiomyocyte2"
    - "Cardiomyocyte3"
  max_cells: 50000

training:
  num_epochs: 3
  learning_rate: 5e-5
  batch_size: 16
```

### 3. Run Training

```bash
# Basic training
python src/main.py

# With evaluation and verbose output
python src/main.py --evaluate --verbose

# Custom config or dataset
python src/main.py --config configs/my_config.yaml --data /path/to/dataset.arrow
```

### Command Line Options

```
--config PATH          Configuration file (default: configs/config.yaml)
--data PATH           Dataset file (overrides config)
--output-dir PATH     Output directory (default: timestamped)
--skip-prepare        Skip data preparation step
--evaluate            Run evaluation after training
--verbose             Show detailed logs
```

## Project Structure

```
geneformer-finetuning/
├── src/
│   ├── main.py                    # Main training script
│   ├── utils.py                   # Utilities
│   ├── data/
│   │   └── dataset_loader.py      # Data loading
│   ├── models/
│   │   └── classifier.py          # Model initialization
│   ├── training/
│   │   └── trainer.py             # Training logic
│   └── evaluation/
│       └── evaluator.py           # Evaluation and plots
├── configs/
│   └── config.yaml                # Training configuration
├── notebooks/
│   └── demo.ipynb                 # Demo notebook
├── README.md
├── requirements.txt
└── .gitignore
```

## Output

Training outputs are saved to `outputs/<timestamp>/`:

```
outputs/20251128_120000/
├── ksplit1/                              # Trained model checkpoint
├── cardiomyocyte_classifier_conf_mat.png # Confusion matrix
└── cardiomyocyte_classifier_predictions_*.png
```

## Configuration Options

| Section | Parameter | Description | Default |
|---------|-----------|-------------|---------|
| `model` | `version` | Model version (V1/V2) | `V1` |
| `data` | `max_cells` | Max cells to use | `50000` |
| `training` | `num_epochs` | Training epochs | `3` |
| `training` | `batch_size` | Batch size | `16` |
| `training` | `learning_rate` | Learning rate | `5e-5` |

## Troubleshooting

### Out of Memory

Reduce batch size and/or max cells in `config.yaml`:

```yaml
training:
  batch_size: 8
  gradient_accumulation_steps: 2
data:
  max_cells: 30000
```

### CUDA Not Available

The code automatically falls back to CPU if CUDA is unavailable.

## Training Time

| GPU | Batch Size | Time (50k cells, 3 epochs) |
|-----|------------|----------------------------|
| T4  | 16         | ~30-45 min                 |
| V100| 32         | ~20-30 min                 |
| A100| 64         | ~10-15 min                 |
| CPU | 4          | ~2-3 hours                 |

## Citation

If using Geneformer in your research, please cite:

```
Theodoris et al. (2023)
Transfer learning enables predictions in network biology
Nature
```

## GitHub Setup

```bash
# Initialize git repository
git init
git add .
git commit -m "Initial commit: Geneformer fine-tuning"

# Create repo on GitHub, then:
git remote add origin https://github.com/yourusername/geneformer-finetuning.git
git branch -M main
git push -u origin main
```

---

**Note**: Make sure large files (models, datasets, outputs) are gitignored before pushing to GitHub.
