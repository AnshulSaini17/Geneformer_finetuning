# 🚀 Google Colab Setup Guide for Geneformer Training

## Why Use Google Colab?

- ✅ **Free GPU access** (NVIDIA T4 with 15GB VRAM)
- ✅ **Faster training** than local CPU/MPS
- ✅ **No local setup required** - everything runs in the cloud
- ✅ **12GB RAM free tier** (upgradeable to 51GB with Colab Pro)

---

## Step-by-Step Setup

### 1. Upload Your Dataset to Google Drive

**You ONLY need to upload your dataset.arrow file - nothing else!**

1. Go to [Google Drive](https://drive.google.com/)
2. Create a new folder structure: `My Drive/Seminar/`
3. Upload your `dataset.arrow` file to this folder
   - Path should be: `My Drive/Seminar/dataset.arrow`

**Note:** 
- The upload may take a while depending on file size and your internet speed
- The Geneformer package and pretrained model will download automatically
- You do NOT need to upload the Geneformer repo or model files

---

### 2. Upload Notebook to Colab

**Option A: Direct Upload**
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click `File` → `Upload notebook`
3. Select your `test.ipynb` file

**Option B: Via Google Drive**
1. Upload `test.ipynb` to your Google Drive
2. Right-click the notebook → `Open with` → `Google Colaboratory`

---

### 3. Enable GPU in Colab

⚠️ **IMPORTANT**: You must enable GPU for fast training!

1. In Colab, click `Runtime` → `Change runtime type`
2. Under "Hardware accelerator", select: **T4 GPU**
   - Free tier: T4 GPU (15GB VRAM)
   - Colab Pro: L4 GPU or A100 (faster options)
3. Click `Save`

---

### 4. Run the Notebook

Run cells in order:

#### **Cell 0: Setup & Mount Drive**
```python
# This cell will:
# - Detect Colab environment
# - Install Geneformer
# - Mount Google Drive
# - Check GPU availability
```

When you run this cell:
- A popup will ask to authorize Google Drive access
- Click the link, sign in, and copy the authorization code
- Paste it back in Colab

#### **Cell 2: Load Dataset**
```python
# This will load your dataset from Google Drive
# Make sure the path matches where you uploaded it
```

#### **Continue with remaining cells** as normal!

---

## Monitoring Your Training

### In the Notebook
Run the **System Monitoring cell** anytime to see:
- CPU usage
- RAM usage  
- GPU memory usage (CUDA)
- Current process stats

### In Colab UI
- Click the **RAM/Disk gauge** in the top-right corner
- Shows real-time RAM and disk usage
- GPU usage shows in the output logs

---

## Important Notes & Tips

### Free Tier Limitations
- **Session timeout**: 12 hours max (Colab Pro: 24 hours)
- **Idle timeout**: 90 minutes of inactivity
- **GPU quota**: Limited daily usage (Colab Pro: more quota)

### Tips to Avoid Interruptions
1. **Save checkpoints**: The notebook is configured to save outputs to Google Drive
2. **Monitor progress**: Check training logs regularly
3. **Keep tab active**: Don't let the tab go idle for too long
4. **Use Colab Pro** for longer training runs (if needed)

### If Training Stops
All outputs are saved to Google Drive at: `My Drive/Seminar/output/`

You can:
1. Resume from the last checkpoint
2. Restart the notebook and continue training

---

## File Paths in Colab vs Local

The notebook automatically detects the environment and uses appropriate paths:

| Location | Dataset Path | Output Path |
|----------|-------------|-------------|
| **Colab** | `/content/drive/MyDrive/Seminar/dataset.arrow` | `/content/drive/MyDrive/Seminar/output/` |
| **Local Mac** | `/Users/anshul/Desktop/Seminar/dataset.arrow` | `/Users/anshul/Desktop/Seminar/output/` |

No need to change paths manually!

---

## Comparing Performance

### Local (M4 Pro)
- ✅ 24GB unified memory
- ✅ Good for development/testing
- ⚠️ Slower for large-scale training
- **Estimated time**: 3-5 hours (50k samples)

### Google Colab (T4 GPU)
- ✅ Dedicated 15GB GPU VRAM
- ✅ Much faster for training
- ✅ Free (with limitations)
- **Estimated time**: 30-60 minutes (50k samples)

---

## Troubleshooting

### "Dataset not found" Error
- Check that `dataset.arrow` is uploaded to `My Drive/Seminar/`
- Verify Google Drive is mounted (run Cell 0)
- Check the path matches exactly

### "Out of Memory" Error
**On Colab:**
- Reduce `max_ncells` to 25000 or less
- Reduce `per_device_train_batch_size` to 4 or 2
- Restart runtime: `Runtime` → `Restart runtime`

**On Local:**
- Reduce `max_ncells` to 10000-25000
- Reduce `per_device_train_batch_size` to 4
- Close other applications

### "GPU not available" Error
- Make sure you selected T4 GPU in Runtime settings
- Restart runtime after changing GPU type
- Check Colab GPU quota (may be exhausted)

### Session Timeout
- Outputs are saved to Google Drive automatically
- Just re-run cells from where you left off
- Consider using Colab Pro for longer sessions

---

## What Gets Downloaded Automatically in Colab?

When you run your notebook in Colab, these will download automatically:

### 1. Geneformer Package (~500MB)
- Installed via: `pip install git+https://huggingface.co/ctheodoris/Geneformer`
- Includes all the code to train and run models
- Takes ~2-3 minutes to install

### 2. Pretrained Model (~1.2GB for V1, ~12GB for V2)
- Downloaded from Hugging Face when training starts
- Model: `ctheodoris/Geneformer` (10M parameters)
- Cached in Colab session (won't re-download if you restart training)
- **First download takes 5-10 minutes**, then it's cached

### 3. Dependencies
- PyTorch, Transformers, and other required packages
- Installed automatically with Geneformer

**Total first-time setup: ~10-15 minutes** (mostly waiting for downloads)

---

## Summary: What You Need to Do

### To Google Drive (Manual Upload):
✅ `dataset.arrow` file (your data)

### Auto-Downloads in Colab (No Action Needed):
- ✅ Geneformer package
- ✅ Pretrained model weights
- ✅ All dependencies (PyTorch, etc.)

### NOT Needed:
- ❌ Geneformer repository folder
- ❌ Pretrained model files (Geneformer-V1-10M folder)
- ❌ Any code files (already in the notebook)

---

## After Training Completes

Your trained model and results will be saved in:
- Colab: `/content/drive/MyDrive/Seminar/output/`
- Local: `/Users/anshul/Desktop/Seminar/output/`

You can download these files to use locally or keep them in Google Drive.

---

## Questions?

The notebook is fully configured to work in both environments. Just:
1. Enable GPU in Colab
2. Upload **ONLY** dataset.arrow to Google Drive
3. Run cells in order (downloads happen automatically)

Happy training! 🎉

