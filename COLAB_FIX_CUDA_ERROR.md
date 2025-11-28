# Fix CUDA Error in Google Colab - Complete Guide

## The Problem
You're getting `CUDA error: device-side assert triggered` even though labels are correct (0, 1, 2).

## Root Cause
The CUDA error happens because your **tokenized data might use a different vocabulary** than the V1 model expects, OR there's a mismatch in model configuration.

---

## Solution: Complete Cell-by-Cell Setup

### Cell 10: Initialize Classifier (FIXED VERSION)

```python
# Define training arguments (matching the example notebook)
training_args = {
    "num_train_epochs": 3,
    "learning_rate": 0.001,
    "lr_scheduler_type": "linear",
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "per_device_train_batch_size": 12,  # Matching example
    "seed": 42,
}

# Filter to only these 3 cardiomyocyte types (matching example)
filter_data_dict = {
    "cell_type": ["Cardiomyocyte1", "Cardiomyocyte2", "Cardiomyocyte3"]
}

# Initialize the Classifier - CRITICAL CHANGES:
cc = Classifier(
    classifier="cell",
    cell_state_dict={"state_key": "cell_type", "states": "all"},
    filter_data=filter_data_dict,  # ← MUST include this
    training_args=training_args,
    max_ncells=50000,
    freeze_layers=2,
    num_crossval_splits=1,
    forward_batch_size=200,
    model_version="V1",  # ← CRITICAL: Must specify V1
    nproc=4
)

print("✓ Classifier initialized!")
print(f"  Filtering to: {filter_data_dict['cell_type']}")
print(f"  Model version: V1")
```

**Key Changes:**
1. ✅ Added `filter_data=filter_data_dict`
2. ✅ Added `model_version="V1"`  ← **This is critical!**
3. ✅ Changed batch_size to 12 (matching example)

---

### Cell 15: Enhanced Debug (ADD THIS NEW CELL)

```python
import os
import numpy as np

# Enable better error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Load and verify prepared data
from datasets import load_from_disk
import pandas as pd

train_file = f"{output_dir}/{output_prefix}_labeled_train.dataset"
train_data = load_from_disk(train_file)

print("="*60)
print("DATA VERIFICATION")
print("="*60)

# Check 1: Labels
labels = pd.Series(train_data['label'])
print(f"\n✓ Labels: {labels.min()} to {labels.max()}")
print(f"  Unique: {sorted(labels.unique().tolist())}")
print(f"  Counts: {labels.value_counts().to_dict()}")

# Check 2: Token IDs (CRITICAL)
print(f"\n✓ Checking token IDs (first 100 samples)...")
max_tokens = [max(train_data[i]['input_ids']) for i in range(min(100, len(train_data)))]
min_tokens = [min(train_data[i]['input_ids']) for i in range(min(100, len(train_data)))]

token_max = max(max_tokens)
token_min = min(min_tokens)

print(f"  Token range: {token_min} to {token_max}")
print(f"  V1 vocab size: ~25,424")

if token_max >= 25424:
    print(f"  ⚠️  ERROR: Tokens exceed V1 vocabulary!")
    print(f"  ⚠️  Max token {token_max} >= vocab size 25,424")
    print(f"  ⚠️  This WILL cause CUDA errors!")
    print(f"\n  Solution: Your data needs to be tokenized with V1 token dictionary")
else:
    print(f"  ✓ Token IDs are valid for V1 model")

# Check 3: Sample data structure
print(f"\n✓ Sample structure:")
sample = train_data[0]
for key in sample.keys():
    val = sample[key]
    if isinstance(val, list):
        print(f"  {key}: list, length={len(val)}, type={type(val[0])}")
    else:
        print(f"  {key}: {val}")

print("\n" + "="*60)
print("✓ Debug checks complete")
print("="*60)
```

**This cell will tell you if the token IDs are the problem!**

---

### Cell 16: Training (UPDATED)

```python
# IMPORTANT: Use V1 model
if IN_COLAB:
    model_directory = "ctheodoris/Geneformer"  # V1 model from HuggingFace
    print(f"✓ Using HuggingFace V1 model")
else:
    model_directory = "/Users/anshul/Desktop/Seminar/Geneformer/Geneformer-V1-10M"
    print(f"✓ Using local V1 model")

# Train the model
print("\nStarting training...")
print("="*60)

all_metrics = cc.validate(
    model_directory=model_directory,
    prepared_input_data_file=f"{output_dir}/{output_prefix}_labeled_train.dataset",
    id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
    output_directory=output_dir,
    output_prefix=output_prefix,
)

print("="*60)
print("✓ Training complete!")
print(f"Metrics: {all_metrics}")
```

---

## Expected Output from Debug Cell

### Good Output (Will Work):
```
DATA VERIFICATION
============================================================

✓ Labels: 0 to 2
  Unique: [0, 1, 2]
  Counts: {0: 15000, 1: 15000, 2: 15000}

✓ Checking token IDs (first 100 samples)...
  Token range: 0 to 23456
  V1 vocab size: ~25,424
  ✓ Token IDs are valid for V1 model

✓ Debug checks complete
```

### Bad Output (Will Fail):
```
✓ Checking token IDs (first 100 samples)...
  Token range: 0 to 106000  ← PROBLEM!
  V1 vocab size: ~25,424
  ⚠️  ERROR: Tokens exceed V1 vocabulary!
  ⚠️  Max token 106000 >= vocab size 25,424
  ⚠️  This WILL cause CUDA errors!
```

---

## If Token IDs Are the Problem

Your dataset was tokenized with the **wrong token dictionary** (probably V2's vocabulary).

### Solution A: Use V2 Model Instead

**Update Cell 10:**
```python
cc = Classifier(
    classifier="cell",
    cell_state_dict={"state_key": "cell_type", "states": "all"},
    filter_data=filter_data_dict,
    training_args=training_args,
    max_ncells=50000,
    freeze_layers=2,
    num_crossval_splits=1,
    forward_batch_size=200,
    model_version="V2",  # ← Changed to V2
    nproc=4
)
```

**Update Cell 16:**
```python
if IN_COLAB:
    model_directory = "ctheodoris/Geneformer-v2-104M"  # ← V2 model
```

### Solution B: Re-tokenize Your Data with V1

You would need to re-tokenize your original data using the V1 token dictionary. This is more work but gives you the V1 model benefits.

---

## Complete Checklist

Before running training in Colab:

- [ ] Cell 10: Includes `model_version="V1"` ← **Critical!**
- [ ] Cell 10: Includes `filter_data=filter_data_dict`
- [ ] Cell 15: Run debug cell - verify token IDs < 25,424
- [ ] Cell 16: Uses matching V1 or V2 model
- [ ] Labels are 0, 1, 2 (verified ✓)
- [ ] 3 classes only (Cardiomyocyte1,2,3) ✓

---

## Quick Fix: Try This First

**In Colab, restart runtime and run:**

1. Cell 0 (Setup)
2. Cell 2 (Load data)
3. **Cell 10 (Initialize with model_version="V1")**
4. Cell 13 (Prepare data)
5. **Cell 15 (Check tokens) ← NEW**
6. Cell 16 (Train)

If Cell 15 shows token IDs > 25,424, **change to model_version="V2"** in Cell 10 and use V2 model in Cell 16.

---

## Still Getting Errors?

Run this in a new cell to check CUDA:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Test simple CUDA operation
if torch.cuda.is_available():
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    y = x * 2
    print(f"✓ Basic CUDA test passed: {y}")
```

If this fails, there's a deeper CUDA/GPU issue with Colab itself.


