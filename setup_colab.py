"""
Setup script for Google Colab environment
Installs all dependencies and mounts Google Drive
"""

import subprocess
import sys
import os


def install_dependencies():
    """Install required packages with correct versions"""
    print("Installing dependencies...")
    
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", 
                   "transformers", "peft", "geneformer"], 
                   capture_output=True)
    
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", 
                   "transformers==4.46.0"], check=True)
    
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", 
                   "peft==0.13.2"], check=True)
    
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                   "datasets", "accelerate", "evaluate", "tdigest",
                   "scikit-learn", "pandas", "numpy", "scanpy", 
                   "loompy", "anndata"], check=True)
    
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                   "git+https://huggingface.co/ctheodoris/Geneformer"], check=True)
    
    print("Installation complete")


def mount_drive():
    """Mount Google Drive"""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive mounted")
        return True
    except:
        print("Not running in Colab, skipping Drive mount")
        return False


def verify_setup():
    """Verify installation"""
    import torch
    import transformers
    import peft
    import geneformer
    
    print("\nEnvironment:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Transformers: {transformers.__version__}")
    print(f"  PEFT: {peft.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    return transformers.__version__ == "4.46.0" and peft.__version__ == "0.13.2"


def main():
    print("="*60)
    print("Geneformer Colab Setup")
    print("="*60)
    
    install_dependencies()
    mount_drive()
    
    if verify_setup():
        print("\nSetup successful. Ready to train.")
    else:
        print("\nWarning: Version mismatch detected")
    
    print("="*60)


if __name__ == "__main__":
    main()


