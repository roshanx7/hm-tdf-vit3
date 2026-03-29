# Complete Dependency Installation Guide for Kaggle

This guide shows all the exact steps to install dependencies for the HM-TDF project on Kaggle.

---

## Step 1: Create Python 3.8 Virtual Environment

```bash
# Check if Python 3.8 is available
python3.8 --version

# If not available, you may need to install it (OS-dependent)
# On Ubuntu/Debian: sudo apt-get install python3.8 python3.8-venv
# On Kaggle: Python 3.8 should be available as `python3.8`

# Create virtual environment
python3.8 -m venv hm_tdf_env

# Activate virtual environment
source hm_tdf_env/bin/activate

# Verify you're in the environment
python --version  # Should show 3.8.x
```

---

## Step 2: Upgrade pip, setuptools, and wheel

```bash
# Must be done AFTER activating venv
source hm_tdf_env/bin/activate

# Upgrade pip to latest version
pip install --upgrade pip

# Install setuptools and wheel (needed for building packages)
pip install setuptools wheel

# Verify installations
pip --version
```

**Expected output:**
```
pip 25.0.1 from .../hm_tdf_env/lib/python3.8/site-packages/pip (python 3.8)
```

---

## Step 3: Install PyTorch (CPU version)

```bash
# IMPORTANT: Use CPU version to avoid GPU/compatibility issues on Kaggle
# This is the most critical dependency

source hm_tdf_env/bin/activate

pip install torch==1.9.0 --index-url https://download.pytorch.org/whl/cpu

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

**Note:** 
- Using CPU version avoids GPU compatibility issues
- PyTorch 1.9.0 is compatible with Python 3.8
- Installation may take 2-5 minutes

---

## Step 4: Install TorchVision

```bash
source hm_tdf_env/bin/activate

pip install torchvision==0.10.0

# Verify
python -c "import torchvision; print(f'TorchVision version: {torchvision.__version__}')"
```

---

## Step 5: Install Other Core Dependencies

```bash
source hm_tdf_env/bin/activate

# Install all other required packages
pip install numpy==1.19.5
pip install pandas==1.1.5
pip install scikit-learn==0.23.2
pip install scipy==1.5.4
pip install pillow==8.4.0
pip install timm==0.9.7
pip install d2l==1.0.3

# Or install all at once:
# pip install numpy==1.19.5 pandas==1.1.5 scikit-learn==0.23.2 scipy==1.5.4 pillow==8.4.0 timm==0.9.7 d2l==1.0.3
```

---

## Step 6: Verify All Installations

```bash
source hm_tdf_env/bin/activate

# Check each package
python -c "
import torch
import torchvision
import numpy
import pandas
import sklearn
import scipy
import PIL
import timm
print('✓ torch:', torch.__version__)
print('✓ torchvision:', torchvision.__version__)
print('✓ numpy:', numpy.__version__)
print('✓ pandas:', pandas.__version__)
print('✓ scikit-learn:', sklearn.__version__)
print('✓ scipy:', scipy.__version__)
print('✓ PIL:', PIL.__version__)
print('✓ timm:', timm.__version__)
print('\n✓ All dependencies installed successfully!')
"
```

**Expected output:**
```
✓ torch: 1.9.0+cpu
✓ torchvision: 0.10.0
✓ numpy: 1.19.5
✓ pandas: 1.1.5
✓ scikit-learn: 0.23.2
✓ scipy: 1.5.4
✓ PIL: 8.4.0
✓ timm: 0.9.7

✓ All dependencies installed successfully!
```

---

## Complete Installation Summary

### All versions pinned (for exact reproducibility)

| Package | Version | Reason |
|---------|---------|--------|
| python | 3.8.x | Main requirement from README; stable with PyTorch 1.9 |
| torch | 1.9.0 (CPU) | Python 3.8 compatible; CPU version for Kaggle |
| torchvision | 0.10.0 | Matches torch 1.9.0; includes ResNet18 |
| numpy | 1.19.5 | Last version supporting Python 3.8 without C extension issues |
| pandas | 1.1.5 | Compatible with numpy 1.19.5 |
| scikit-learn | 0.23.2 | Stable version for Python 3.8 |
| scipy | 1.5.4 | Compatible with numpy 1.19.5 |
| pillow | 8.4.0 | Stable image processing library |
| timm | 0.9.7 | Provides ResNet18 (TinyViT had bugs with Python 3.8) |
| d2l | 1.0.3 | Deep learning utilities |

---

## One-Liner Installation (Fastest Method)

If you want to install everything at once:

```bash
python3.8 -m venv hm_tdf_env && \
source hm_tdf_env/bin/activate && \
pip install --upgrade pip setuptools wheel && \
pip install torch==1.9.0 --index-url https://download.pytorch.org/whl/cpu torchvision==0.10.0 numpy==1.19.5 pandas==1.1.5 scikit-learn==0.23.2 scipy==1.5.4 pillow==8.4.0 timm==0.9.7 d2l==1.0.3
```

---

## For Kaggle Notebooks

If running on Kaggle Notebooks:

```python
# In a Kaggle notebook cell:
!python3.8 -m venv /tmp/hm_tdf_env
!source /tmp/hm_tdf_env/bin/activate && pip install --upgrade pip setuptools wheel
!source /tmp/hm_tdf_env/bin/activate && pip install torch==1.9.0 --index-url https://download.pytorch.org/whl/cpu torchvision==0.10.0 numpy==1.19.5 pandas==1.1.5 scikit-learn==0.23.2 scipy==1.5.4 pillow==8.4.0 timm==0.9.7 d2l==1.0.3

# Then in subsequent cells:
import sys
sys.path.insert(0, '/tmp/hm_tdf_env/lib/python3.8/site-packages')
```

Or use requirements.txt (corrected version):

```bash
# File: requirements.txt
numpy==1.19.5
pandas==1.1.5
scikit-learn==0.23.2
scipy==1.5.4
torch==1.9.0
torchvision==0.10.0
pillow==8.4.0
d2l==1.0.3
timm==0.9.7
```

Then:
```bash
python3.8 -m venv hm_tdf_env
source hm_tdf_env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt  # Won't work with torch directly; use the pip commands above instead
```

---

## Troubleshooting on Kaggle

### Issue 1: Python 3.8 Not Found
```bash
# Check available Python versions
ls /usr/bin/python*

# If only 3.9+ available, you may need to adjust versions
# Contact Kaggle support or use official kernel with Python 3.8
```

### Issue 2: Cache Issues
```bash
# Clear pip cache before installing
pip cache purge

# Then retry installation
pip install torch==1.9.0 --index-url https://download.pytorch.org/whl/cpu
```

### Issue 3: Memory Issues During Installation
```bash
# Install one package at a time instead of all at once
pip install torch==1.9.0 --index-url https://download.pytorch.org/whl/cpu
pip install torchvision==0.10.0
pip install numpy==1.19.5
# ... etc
```

### Issue 4: Network Timeout
```bash
# Increase timeout for pip
pip install --default-timeout=1000 torch==1.9.0 --index-url https://download.pytorch.org/whl/cpu
```

---

## Verify Model Works After Installation

```bash
source hm_tdf_env/bin/activate
cd /path/to/HM-TDF

# Download or copy the project files to Kaggle

# Run test
python test_modified_model.py
```

**Expected output:**
```
✓ Successfully imported models.Mffkan
✓ Successfully created model
✓ Forward pass successful
✓ All tests passed!
```

---

## Running the Training Pipeline on Kaggle

After installation is complete:

```bash
source hm_tdf_env/bin/activate

# 1. Extract images
cd ./Tongue-FLD
cat Tongue_Images.tar.gz.* > Tongue_Images.tar.gz
tar xzf Tongue_Images.tar.gz
cd ..

# 2. Generate augmented images
python random_rotate_images.py

# 3. Pre-train on rotated images
python rotate_pre_train.py

# 4. Main training
python main.py
```

---

## Key Differences from Local Installation

| Aspect | Local | Kaggle |
|--------|-------|--------|
| Python Install | May need apt/brew | Usually pre-installed |
| PyTorch | Can use GPU or CPU | Use CPU (`--index-url https://download.pytorch.org/whl/cpu`) |
| Storage | Persistent | May have limits (check Kaggle) |
| Temp Files | Can use /tmp | OK to use /tmp |
| Venv Location | Can be anywhere | Recommended: `/tmp/` or project root |
| Session Duration | Persistent | 9 hours max per session |

---

## Final Checklist

- [ ] Python 3.8 available (`python3.8 --version`)
- [ ] Virtual environment created (`source hm_tdf_env/bin/activate`)
- [ ] pip upgraded
- [ ] setuptools and wheel installed
- [ ] PyTorch 1.9.0 CPU installed
- [ ] TorchVision 0.10.0 installed
- [ ] All other dependencies installed
- [ ] Verification script runs successfully
- [ ] test_modified_model.py passes all tests

---

## Notes

1. **CPU-only PyTorch:** Using `--index-url https://download.pytorch.org/whl/cpu` ensures compatibility and avoids GPU-related issues

2. **Pinned Versions:** All versions are specifically pinned because:
   - Python 3.8 is EOL, so only specific package versions work
   - These versions have been tested together
   - Newer versions may have incompatibilities

3. **Installation Time:** Expect 5-15 minutes total for all installations depending on Kaggle's internet speed

4. **Disk Space:** Required ~2-3 GB for all packages

