# AnCore - Quick Start Guide

## Welcome to AnCore! 🎯

This guide will help you get started with the AnCore fake news detection system.

## 📋 Prerequisites Check

Before starting, make sure you have:
- ✅ Python 3.8 or higher installed
- ✅ pip package manager
- ✅ Data file: `fakenews/full.csv` in the project directory

## 🚀 Step-by-Step Setup

### Step 1: Install Dependencies

Open PowerShell in the project directory and run:

```powershell
pip install -r requirements.txt
```

This will install all required packages including:
- PyTorch (deep learning framework)
- Transformers (for mBERT model)
- Pandas (data processing)
- Scikit-learn (evaluation metrics)
- Matplotlib & Seaborn (visualization)

**Note:** Installation may take 5-10 minutes depending on your internet connection.

### Step 2: Verify Setup

Run the demo script to verify everything is working:

```powershell
python demo.py
```

This will:
- Check if the data file exists
- Load a sample of the dataset
- Initialize the mBERT model
- Display configuration settings

### Step 3: View Sample Data (Optional)

To see examples of real and fake news from the dataset:

```powershell
python demo.py --samples
```

## 🎓 Training the Model

### Quick Training (Recommended for First Time)

For a quick test with default settings:

```powershell
python ancore_main.py --mode train
```

**What happens during training:**
1. Data is split into train (80%), validation (10%), and test (10%)
2. Model trains for 5 epochs (adjustable in config)
3. Best model is saved automatically
4. Training curves are plotted
5. Final evaluation on test set
6. Confusion matrix is generated

**Expected duration:** 
- With GPU: 15-30 minutes
- With CPU: 1-2 hours

### Monitoring Progress

During training, you'll see:
```
Epoch 1/5
--------------------------------------------------
Training: 100%|████████████| 160/160 [05:23<00:00]
Validation: 100%|██████████| 20/20 [00:34<00:00]
Train Loss: 0.4523 | Train Acc: 0.7845
Val Loss: 0.3821 | Val Acc: 0.8234
✓ New best model saved!
```

## 🔍 Using the Trained Model

### Interactive Mode (Most User-Friendly)

Launch interactive mode to test articles:

```powershell
python ancore_main.py --mode interactive
```

Then enter articles to analyze:
```
Enter article text (or 'quit' to exit):
> [Your article here]

============================================================
CREDIBILITY ASSESSMENT
============================================================
Prediction: Real News
Confidence: 92.34% (High)

Probability Breakdown:
  Real News: 92.34%
  Fake News: 7.66%

Credibility Score: 92.3/100
============================================================
```

### Single Article Prediction

Predict a single article via command line:

```powershell
python ancore_main.py --mode predict --text "Your news article text here"
```

### Evaluate Model Performance

Check model performance on test set:

```powershell
python ancore_main.py --mode evaluate
```

## 📊 Understanding the Output

### Training Output Files

After training, check the `output/` directory:

```
output/
├── models/
│   └── best_model.pt              # Best trained model
├── results/
│   ├── training_history.png       # Training curves
│   ├── confusion_matrix.png       # Confusion matrix
│   └── results_YYYYMMDD.json     # Detailed metrics
└── logs/                          # Training logs
```

### Performance Metrics

The model provides several metrics:

- **Accuracy**: Overall correctness (target: >85%)
- **Precision**: How many predicted fake news are actually fake
- **Recall**: How many actual fake news were detected
- **F1-Score**: Balance between precision and recall
- **Confidence**: How certain the model is about its prediction

### Credibility Levels

- **High Confidence** (≥80%): Very reliable prediction
- **Medium Confidence** (50-80%): Moderately reliable
- **Low Confidence** (<50%): Uncertain prediction

## ⚙️ Customization

### Adjust Training Parameters

Edit `ancore_config.py` to customize:

```python
# Training speed vs accuracy
BATCH_SIZE = 16          # Smaller = slower but more stable
LEARNING_RATE = 2e-5     # Lower = slower but more precise
NUM_EPOCHS = 5           # More epochs = better but longer

# Memory management
MAX_LENGTH = 512         # Shorter = less memory, faster
```

### Common Adjustments

**For faster training (lower accuracy):**
```python
BATCH_SIZE = 32
NUM_EPOCHS = 3
MAX_LENGTH = 256
```

**For better accuracy (slower training):**
```python
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 1e-5
```

## 🐛 Troubleshooting

### Problem: "CUDA out of memory"

**Solution:** Reduce batch size or max length:
```python
BATCH_SIZE = 8
MAX_LENGTH = 256
```

### Problem: "Data file not found"

**Solution:** Ensure `fakenews/full.csv` exists:
```powershell
dir fakenews\full.csv
```

### Problem: Training is very slow

**Solutions:**
1. Check if GPU is available: `torch.cuda.is_available()`
2. Reduce `NUM_EPOCHS` to 3 for testing
3. Reduce `MAX_LENGTH` to 256
4. Close other applications

### Problem: Import errors

**Solution:** Reinstall requirements:
```powershell
pip install -r requirements.txt --upgrade
```

## 📈 Expected Performance

Based on the dataset:
- **Training Accuracy**: 85-95%
- **Validation Accuracy**: 80-90%
- **Test Accuracy**: 80-90%
- **F1-Score**: 0.80-0.90

Lower performance may indicate:
- Need more training epochs
- Overfitting (if train >> test accuracy)
- Need to adjust learning rate

## 💡 Tips for Best Results

1. **First Run**: Use default settings to establish baseline
2. **GPU Usage**: Enable CUDA for 10x faster training
3. **Data Quality**: Clean, balanced dataset is crucial
4. **Monitoring**: Watch validation accuracy to prevent overfitting
5. **Patience**: Good models take time to train

## 🎯 Workflow Examples

### Example 1: Quick Evaluation

```powershell
# 1. Train with defaults
python ancore_main.py --mode train

# 2. Test with interactive mode
python ancore_main.py --mode interactive
```

### Example 2: Production Use

```powershell
# 1. Train with optimized settings (edit config first)
python ancore_main.py --mode train

# 2. Evaluate thoroughly
python ancore_main.py --mode evaluate

# 3. Use for predictions
python ancore_main.py --mode predict --text "Article text"
```

### Example 3: Research & Development

```powershell
# 1. Run demo to verify setup
python demo.py

# 2. View sample articles
python demo.py --samples

# 3. Train model
python ancore_main.py --mode train

# 4. Analyze results in output/results/
```

## 📚 Next Steps

After successful setup:

1. ✅ Train your first model
2. ✅ Test with interactive mode
3. ✅ Review training curves
4. ✅ Experiment with different articles
5. ✅ Adjust parameters for better performance

## 🆘 Getting Help

If you encounter issues:

1. Check error messages carefully
2. Review the troubleshooting section
3. Verify all files are in place
4. Ensure dependencies are installed
5. Check Python version (≥3.8)

## 🎉 Success Indicators

You're ready when you see:

- ✅ Model trains without errors
- ✅ Validation accuracy > 80%
- ✅ Interactive mode works
- ✅ Plots are generated
- ✅ Model checkpoint saved

---

**Happy fake news hunting! 🚀**

For detailed documentation, see `README.md`
