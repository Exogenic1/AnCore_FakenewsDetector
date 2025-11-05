# AnCore - Troubleshooting Guide

## Common Issues and Solutions

### ✅ SOLVED: NumPy 2.x Compatibility Issue

**Problem:**
```
AttributeError: _ARRAY_API not found
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
```

**Cause:**
The new NumPy 2.x is incompatible with older versions of TensorFlow, Keras, matplotlib, and cv2 that were compiled against NumPy 1.x.

**Solution:**
```powershell
pip install "numpy<2.0" --force-reinstall
```

Or use the Python package installer:
```powershell
python -m pip install "numpy<2.0" --force-reinstall
```

**Prevention:**
The `requirements.txt` has been updated to specify `numpy>=1.24.0,<2.0` to prevent this issue in future installations.

---

### 🔧 Other Common Issues

#### 1. CUDA Not Available

**Symptom:**
```
✗ CUDA GPU: FAILED - Not available (will use CPU)
```

**Solution:**
This is normal if you don't have an NVIDIA GPU. The system will use CPU instead.
- Training will take longer (1-2 hours vs 15-30 minutes)
- Everything will still work correctly

**To enable CUDA (if you have NVIDIA GPU):**
1. Install CUDA Toolkit from NVIDIA website
2. Install PyTorch with CUDA support:
   ```powershell
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

---

#### 2. Out of Memory Error

**Symptom:**
```
RuntimeError: CUDA out of memory
```
or
```
MemoryError
```

**Solution:**
Reduce memory usage by editing `ancore_config.py`:

```python
# Reduce batch size
BATCH_SIZE = 8  # or even 4

# Reduce sequence length
MAX_LENGTH = 256  # from 512
```

---

#### 3. Transformers Package Issues

**Symptom:**
```
ModuleNotFoundError: No module named 'transformers'
```

**Solution:**
```powershell
pip install transformers>=4.30.0
```

If still failing:
```powershell
pip install --upgrade transformers
```

---

#### 4. Data File Not Found

**Symptom:**
```
FileNotFoundError: Data file not found at fakenews/full.csv
```

**Solution:**
1. Verify the file exists:
   ```powershell
   dir fakenews\full.csv
   ```

2. If missing, ensure you have the correct directory structure:
   ```
   Fakenews/
   ├── fakenews/
   │   └── full.csv
   └── [other files]
   ```

---

#### 5. Slow Import Times (TensorFlow warnings)

**Symptom:**
```
oneDNN custom operations are on. You may see slightly different numerical results...
```

**Solution:**
This is just a warning and can be safely ignored. It's TensorFlow optimizing performance.

To suppress the warning, set environment variable:
```powershell
$env:TF_ENABLE_ONEDNN_OPTS="0"
```

---

#### 6. Package Dependency Conflicts

**Symptom:**
```
ERROR: pip's dependency resolver does not currently take into account all packages
```

**Solution:**
1. Create a fresh virtual environment:
   ```powershell
   python -m venv ancore_env
   .\ancore_env\Scripts\Activate.ps1
   ```

2. Install requirements:
   ```powershell
   pip install -r requirements.txt
   ```

---

#### 7. Training Too Slow

**Symptom:**
Training takes more than 2 hours

**Solutions:**

**Option 1: Reduce epochs for testing**
Edit `ancore_config.py`:
```python
NUM_EPOCHS = 3  # instead of 5
```

**Option 2: Use smaller dataset for testing**
Edit `ancore_dataset.py` to sample the data:
```python
df = df.sample(frac=0.5)  # Use 50% of data
```

**Option 3: Reduce model complexity**
Edit `ancore_config.py`:
```python
BATCH_SIZE = 32  # Larger batches = fewer iterations
MAX_LENGTH = 256  # Shorter sequences = faster processing
```

---

#### 8. Model Download Fails

**Symptom:**
```
ConnectionError: Can't reach https://huggingface.co
```

**Solution:**
1. Check internet connection
2. Try again (might be temporary)
3. Use offline mode if you've downloaded before:
   ```python
   from transformers import BertModel
   model = BertModel.from_pretrained('bert-base-multilingual-cased', local_files_only=True)
   ```

---

#### 9. Matplotlib Backend Issues

**Symptom:**
```
UserWarning: Matplotlib is currently using agg, which is a non-GUI backend
```

**Solution:**
This is normal for systems without display. Plots will still be saved.

To view plots, use:
```python
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
```

---

#### 10. Permission Denied Errors

**Symptom:**
```
PermissionError: [Errno 13] Permission denied: 'output/models/best_model.pt'
```

**Solution:**
1. Close any programs that might be using the files
2. Run PowerShell as Administrator
3. Check file permissions:
   ```powershell
   icacls output\models\best_model.pt
   ```

---

## Quick Fixes Summary

| Issue | Quick Fix |
|-------|-----------|
| NumPy 2.x error | `pip install "numpy<2.0" --force-reinstall` |
| Out of memory | Reduce `BATCH_SIZE` and `MAX_LENGTH` |
| Slow training | Reduce `NUM_EPOCHS` or use smaller dataset |
| Missing packages | `pip install -r requirements.txt` |
| Data not found | Check `fakenews/full.csv` exists |
| No GPU | Normal on CPU-only systems, training will work |

---

## System Requirements

### Minimum
- Python 3.8+
- 8 GB RAM
- 5 GB free disk space
- CPU (any modern processor)

### Recommended
- Python 3.10+
- 16 GB RAM
- 10 GB free disk space
- NVIDIA GPU with CUDA support

---

## Getting Help

1. Run verification script:
   ```powershell
   python verify_setup.py
   ```

2. Check all packages are installed:
   ```powershell
   pip list | Select-String "torch|transformers|numpy|pandas"
   ```

3. Test with demo:
   ```powershell
   python demo.py
   ```

4. Check Python version:
   ```powershell
   python --version
   ```

---

## Useful Commands

### Reinstall All Dependencies
```powershell
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

### Clear Package Cache
```powershell
pip cache purge
```

### Check Package Versions
```powershell
pip show torch transformers numpy
```

### Test Imports
```powershell
python -c "import torch; import transformers; print('OK')"
```

---

## Contact & Support

If you encounter issues not covered here:

1. Check the error message carefully
2. Review the relevant section above
3. Try the verification script: `python verify_setup.py`
4. Ensure all requirements are installed: `pip install -r requirements.txt`

---

**Last Updated:** November 4, 2025  
**Status:** All known issues resolved ✅
