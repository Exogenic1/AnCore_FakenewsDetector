# AnCore Web Application - User Guide

## 🌐 Web Interface for Fake News Detection

This guide will help you launch and use the AnCore web application - a user-friendly interface for detecting fake news in Filipino articles.

---

## 📋 Prerequisites

Before launching the web app, ensure:

1. ✅ You have the trained model at `output/models/best_model.pt`
2. ✅ All dependencies are installed (including Streamlit)
3. ✅ Python 3.8+ is installed

---

## 🚀 Quick Start

### Step 1: Install Streamlit (if not already installed)

```powershell
pip install streamlit
```

Or install all dependencies:

```powershell
pip install -r requirements.txt
```

### Step 2: Launch the Web Application

```powershell
streamlit run web_app.py
```

The application will automatically open in your default web browser at:
- **Local URL**: http://localhost:8501
- **Network URL**: http://[your-ip]:8501

### Step 3: Use the Application

1. **Enter Article**: Paste or type a Filipino news article in the text box
2. **Click "Analyze Article"**: Press the green button to check the article
3. **Review Results**: See the prediction, confidence score, and detailed metrics
4. **Check History**: View your previous analyses in the history section

---

## 🎨 Features

### 📝 **Easy Text Input**
- Large text area for pasting articles
- Character and word count
- Sample articles included for testing
- Real-time validation

### 🎯 **Clear Results Display**
- **Color-coded predictions**: Green for Real, Red for Fake
- **Confidence levels**: High/Medium/Low
- **Credibility score**: 0-100 scale
- **Probability breakdown**: Real vs Fake percentages

### 📊 **Visual Analytics**
- Progress bars for probability visualization
- Metric cards with key statistics
- Professional design with intuitive layout

### 📜 **Analysis History**
- Track all your analyses
- Timestamp for each check
- Quick reference to previous results
- Clear history option

### 💡 **Helpful Guidance**
- Interpretation guide for results
- Tips on how to use the tool
- Warnings for short texts
- Model information in sidebar

---

## 🖥️ User Interface Overview

### Main Screen

```
┌─────────────────────────────────────────────┐
│   🔍 AnCore - Fake News Detector           │
│   AI-Powered News Credibility Assessment   │
├─────────────────────────────────────────────┤
│                                             │
│   📝 Enter News Article                    │
│   ┌───────────────────────────────────┐   │
│   │                                   │   │
│   │  [Text Area - Paste Article]     │   │
│   │                                   │   │
│   └───────────────────────────────────┘   │
│                                             │
│   Characters: 450  Words: 85  Status: ✅   │
│                                             │
│   [🔍 Analyze Article]                     │
│                                             │
├─────────────────────────────────────────────┤
│   📊 Analysis Results                      │
│                                             │
│   ✅ Real News                             │
│   Confidence: 92.3% (High)                 │
│                                             │
│   Credibility  Real News   Fake News       │
│      92.3        92.3%       7.7%          │
│                                             │
│   💡 How to Interpret the Results          │
│   High Confidence Prediction...            │
└─────────────────────────────────────────────┘
```

### Sidebar

```
┌──────────────────────────┐
│ 📖 About AnCore         │
│                         │
│ 🎯 How to Use          │
│ 1. Paste article       │
│ 2. Click Analyze       │
│ 3. Review results      │
│                         │
│ ⚙️ Model Info          │
│ • mBERT Model          │
│ • 177M parameters      │
│                         │
│ ⚠️ Important Notes     │
│                         │
│ 📊 Quick Stats         │
│ Accuracy: ~85-90%      │
└──────────────────────────┘
```

---

## 💡 How to Use (Step-by-Step)

### For First-Time Users

1. **Launch the App**
   ```powershell
   streamlit run web_app.py
   ```
   - Wait for browser to open automatically
   - If not, open http://localhost:8501 manually

2. **Try a Sample Article**
   - Select "Sample Real News" or "Sample Fake News" from dropdown
   - Click "Analyze Article"
   - Observe the results

3. **Test Your Own Article**
   - Select "Type Your Own" from dropdown
   - Paste your Filipino news article
   - Ensure at least 20 words for best accuracy
   - Click "Analyze Article"

4. **Interpret Results**
   - **Green = Real News**: The AI thinks it's legitimate
   - **Red = Fake News**: The AI detects potential misinformation
   - **Confidence**: How sure the AI is (higher = more certain)
   - **Credibility Score**: 0-100 rating (higher = more credible)

### Understanding Confidence Levels

| Confidence | Meaning | What to Do |
|------------|---------|------------|
| **High (≥80%)** | Very confident prediction | Can trust result, but still verify |
| **Medium (50-80%)** | Moderately confident | Recommend additional verification |
| **Low (<50%)** | Uncertain prediction | Definitely verify from other sources |

---

## 🎯 Tips for Best Results

### ✅ DO:
- Use **complete articles** (not just headlines)
- Paste **at least 20 words** for accuracy
- Include the **full context** of the news
- Try **multiple similar articles** to compare

### ❌ DON'T:
- Don't use single sentences or headlines only
- Don't rely solely on this tool for important decisions
- Don't assume 100% accuracy - it's AI, not perfect
- Don't test with non-Filipino text (model trained on Filipino)

---

## 🔧 Troubleshooting

### App Won't Start

**Problem**: `streamlit: command not found`

**Solution**:
```powershell
pip install streamlit
```

---

### Model Not Found Error

**Problem**: `Model file not found at output/models/best_model.pt`

**Solution**: Train the model first:
```powershell
python ancore_main.py --mode train
```

---

### Slow Performance

**Problem**: Analysis takes more than 10 seconds

**Solutions**:
- First analysis is always slower (model loading)
- Subsequent analyses are much faster (cached)
- Close other heavy applications
- Restart the app if it becomes sluggish

---

### Browser Doesn't Open

**Problem**: App runs but browser doesn't open

**Solution**: Manually open:
```
http://localhost:8501
```

---

### Port Already in Use

**Problem**: `Address already in use`

**Solution**: Use a different port:
```powershell
streamlit run web_app.py --server.port 8502
```

---

## 📱 Accessing from Other Devices

### On Same Network

1. Find your computer's IP address:
   ```powershell
   ipconfig
   ```
   Look for "IPv4 Address" (e.g., 192.168.1.100)

2. On other device, open browser and go to:
   ```
   http://[your-ip]:8501
   ```
   Example: `http://192.168.1.100:8501`

3. Both devices must be on the same WiFi network

---

## ⚙️ Advanced Options

### Custom Port

```powershell
streamlit run web_app.py --server.port 8080
```

### Run in Background (Headless)

```powershell
streamlit run web_app.py --server.headless true
```

### Custom Theme

Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#4CAF50"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

---

## 📊 Sample Usage Scenarios

### Scenario 1: Checking Social Media Posts

**Use Case**: You saw a viral news post on Facebook

**Steps**:
1. Copy the article text from Facebook
2. Paste into AnCore web app
3. Click "Analyze Article"
4. Check the credibility score
5. If score is low or marked as fake, verify from official news sources

---

### Scenario 2: Student Research

**Use Case**: Student verifying sources for academic paper

**Steps**:
1. Paste each source article into the app
2. Record the credibility scores
3. Use only articles with high credibility (≥80%)
4. Cross-reference with multiple sources
5. Keep analysis history for reference

---

### Scenario 3: Community Fact-Checking

**Use Case**: Community moderator checking shared articles

**Steps**:
1. Collect articles shared in the community
2. Analyze each one using the web app
3. Flag articles marked as "Fake News" with high confidence
4. Provide warnings to community members
5. Maintain history for transparency

---

## 🎨 UI Features Explained

### Color Coding

- 🟢 **Green**: Real News - Safe, credible
- 🔴 **Red**: Fake News - Potentially misleading
- 🟡 **Yellow**: Low confidence - Needs verification

### Progress Bars

- Visual representation of probabilities
- Easy to understand at a glance
- Compare Real vs Fake percentages

### Metric Cards

- **Credibility Score**: Overall trustworthiness (0-100)
- **Real News %**: Likelihood of being real
- **Fake News %**: Likelihood of being fake

---

## 🔒 Privacy & Security

- ✅ **All processing is local** - No data sent to external servers
- ✅ **Your articles are not stored** - Only session history (temporary)
- ✅ **Clear history anytime** - One-click cleanup
- ✅ **No login required** - Anonymous usage
- ✅ **No tracking** - Your privacy is protected

---

## 🆘 Getting Help

### Common Questions

**Q: How accurate is the detector?**  
A: Approximately 85-90% accurate, but always verify important news.

**Q: Can it detect all fake news?**  
A: No AI is perfect. Use it as a helpful tool, not the final authority.

**Q: Does it work with English articles?**  
A: It's trained on Filipino articles and works best with Filipino/Tagalog text.

**Q: Can I use it offline?**  
A: Yes! Once the model is loaded, it works without internet.

**Q: How long does analysis take?**  
A: Usually 2-5 seconds after the first load.

---

## 📞 Support

If you encounter issues:

1. Check this guide's troubleshooting section
2. Verify model is trained: `dir output\models\best_model.pt`
3. Check Streamlit is installed: `pip show streamlit`
4. Restart the application

---

## 🎉 Quick Reference Card

```
┌────────────────────────────────────────┐
│  ANCORE WEB APP - QUICK REFERENCE     │
├────────────────────────────────────────┤
│  START:   streamlit run web_app.py    │
│  ACCESS:  http://localhost:8501       │
│  STOP:    Press Ctrl+C in terminal    │
├────────────────────────────────────────┤
│  ✅ Real News    = Green, High Score  │
│  ⚠️ Fake News    = Red, Low Score     │
│  🟡 Uncertain   = Yellow, Medium       │
├────────────────────────────────────────┤
│  TIPS:                                 │
│  • Use complete articles (20+ words)  │
│  • Check confidence level             │
│  • Verify important news elsewhere    │
│  • View history for comparison        │
└────────────────────────────────────────┘
```

---

**Enjoy using AnCore! Help fight fake news! 🚀**
