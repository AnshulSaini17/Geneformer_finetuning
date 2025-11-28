# Geneformer Cell Type Classification

Fine-tuning Geneformer V1 for cardiomyocyte subtype classification.

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- Google Colab with T4/L4 GPU (or local NVIDIA GPU)

## Quick Start (Google Colab)

### 1. Setup Environment

```python
!git clone <your-repo-url>
%cd <repo-name>
!python setup_colab.py
```

### 2. Upload Data

Upload `dataset.arrow` to `Google Drive/Seminar/`

### 3. Run Training

```python
!python train_geneformer.py
```

## Configuration

Edit `train_geneformer.py` to modify:

- `MAX_CELLS`: Number of cells to use (default: 50000)
- `CELL_TYPES`: Cell types to classify
- `TRAINING_CONFIG`: Training hyperparameters

## Output

Results saved to `Google Drive/Seminar/output/<timestamp>/`:

- `ksplit1/`: Trained model checkpoint
- `*_predictions.pkl`: Model predictions
- `*_conf_mat.png`: Confusion matrix
- `*_predictions_*.png`: Prediction visualizations

## Training Time

- **GPU (T4)**: ~30-45 minutes
- **CPU**: ~2-3 hours

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 5e-5 | Initial learning rate |
| Batch Size | 4 | Per-device batch size |
| Epochs | 3 | Number of training epochs |
| Warmup Steps | 500 | Linear warmup steps |
| Weight Decay | 0.001 | L2 regularization |

## Troubleshooting

### CUDA Errors

If you encounter CUDA device-side assert errors, try:

```python
# Reduce batch size
TRAINING_CONFIG["per_device_train_batch_size"] = 1
TRAINING_CONFIG["gradient_accumulation_steps"] = 4
```

### Out of Memory

```python
# Reduce number of cells
MAX_CELLS = 30000
```

### CPU Mode

```python
# Add to setup_colab.py or start of train_geneformer.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

## Model Info

- **Base Model**: Geneformer V1 (10M parameters)
- **Architecture**: BERT-based transformer
- **Vocabulary Size**: ~25,000 genes
- **Max Sequence Length**: 2048 tokens

## Citation

If using Geneformer, cite:

```
Theodoris et al. (2023)
Transfer learning enables predictions in network biology
Nature
```


