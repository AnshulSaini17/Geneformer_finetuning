# Data Preparation Guide

## ⚠️ Data Not Included

This repository does **not include training data** due to:
- Large file sizes (multi-GB)
- Potential privacy/licensing restrictions
- Data may be institution-specific

## 📊 Data Requirements

Your dataset should be a **tokenized Arrow file** with the following structure:

```python
{
    'input_ids': List[int],      # Tokenized gene expression (required)
    'length': int,                # Sequence length (required)
    'cell_type': str,             # Cell type label (required for classification)
    # Optional metadata:
    'individual': str,
    'disease': str,
    'age': float,
    'sex': str,
    # ... any other metadata columns
}
```

## 🔧 How to Get Data

### Option 1: Tokenize Your Own Single-Cell Data

If you have raw single-cell RNA-seq data (.loom, .h5ad, etc.):

```python
from geneformer import TranscriptomeTokenizer

# Initialize tokenizer
tk = TranscriptomeTokenizer(
    custom_attr_name_dict={"cell_type": "cell_type"},
    nproc=4  # Number of processes
)

# Tokenize your data
tk.tokenize_data(
    data_directory="path/to/your/loom_or_h5ad_files/",
    output_directory="output/",
    output_prefix="tokenized_data",
    file_format="loom"  # or "h5ad"
)
```

**See:** [Geneformer Tokenization Example](https://huggingface.co/ctheodoris/Geneformer/blob/main/examples/tokenizing_scRNAseq_data.ipynb)

### Option 2: Use Public Datasets

#### A. Genecorpus-30M (Geneformer Pre-training Data)
- **Source:** [HuggingFace - Geneformer](https://huggingface.co/ctheodoris/Geneformer)
- **Size:** ~30M cells from 27M+ single-cell transcriptomes
- **Format:** Already tokenized
- **Note:** Large download (~100GB+)

#### B. CellxGene Portal
- **Source:** [cellxgene.cziscience.com](https://cellxgene.cziscience.com/)
- **Contains:** Millions of single-cell transcriptomes
- **Format:** .h5ad (needs tokenization)
- **License:** Varies by dataset

#### C. Your Institution's Data
- Contact your research group/institution
- May have existing tokenized datasets
- Often specific to your research domain

### Option 3: Use a Small Test Dataset

For testing the code without full data:

```python
from datasets import Dataset
import random

# Create a small dummy dataset
n_samples = 1000
dummy_data = {
    'input_ids': [[random.randint(0, 25000) for _ in range(2048)] for _ in range(n_samples)],
    'length': [2048] * n_samples,
    'cell_type': random.choices(['Cardiomyocyte1', 'Cardiomyocyte2', 'Cardiomyocyte3'], k=n_samples),
}

dataset = Dataset.from_dict(dummy_data)
dataset.save_to_disk("test_dataset.arrow")
```

**Note:** Dummy data won't train a useful model but lets you test the code!

## 📁 Expected File Structure

After preparing your data:

```
your-project/
├── dataset.arrow              # Your tokenized dataset
└── Geneformer_finetuning/    # This repo
    ├── src/
    ├── configs/
    │   └── config.yaml         # Update dataset_file path here
    └── ...
```

Update `configs/config.yaml`:

```yaml
data:
  dataset_file: "../dataset.arrow"  # or absolute path
  cell_types:
    - "YourCellType1"
    - "YourCellType2"
    # ... your specific cell types
```

## 🚀 Quick Start with Your Data

```bash
# 1. Prepare your tokenized dataset (dataset.arrow)

# 2. Update config
nano configs/config.yaml  # Set dataset path and cell types

# 3. Run training
python src/main.py --evaluate --verbose
```

## 📚 Additional Resources

- **Geneformer Paper:** [Theodoris et al., Nature 2023](https://www.nature.com/articles/s41586-023-06139-9)
- **Geneformer Docs:** [HuggingFace Documentation](https://huggingface.co/ctheodoris/Geneformer)
- **Example Notebooks:** See `Geneformer/examples/` in the official repo
- **Single-cell Analysis:** [Scanpy Tutorials](https://scanpy.readthedocs.io/)

## ❓ FAQ

**Q: Can I use your exact dataset?**  
A: The dataset used in development may be private/restricted. Follow the options above to prepare your own data.

**Q: What if my data is in a different format?**  
A: Use Geneformer's `TranscriptomeTokenizer` to convert .loom, .h5ad, or other formats to the required tokenized format.

**Q: How much data do I need?**  
A: For fine-tuning, 10,000-100,000+ cells is typical. More data generally gives better results.

**Q: What about data privacy?**  
A: Never upload patient data or restricted datasets to public repositories. Always follow your institution's data policies.

## 💡 Tips

1. **Start small** - Test with 10k cells before scaling up
2. **Check cell types** - Ensure your labels match your research question
3. **Validate tokenization** - Verify `input_ids` look reasonable
4. **Balance classes** - Try to have similar numbers of each cell type
5. **Document provenance** - Keep track of where your data came from

---

**Need help?** Open an issue on GitHub with details about your data format and we can assist!

