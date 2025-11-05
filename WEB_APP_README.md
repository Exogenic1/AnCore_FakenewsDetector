# 🌐 AnCore Web Application

## Quick Start - For Non-Technical Users

### 🚀 **Easiest Way to Launch**

**Option 1: Double-Click Launcher (Windows)**
1. Find the file: `launch_web_app.bat`
2. Double-click it
3. Wait for your browser to open
4. Start checking articles!

**Option 2: Using Command Line**
```powershell
streamlit run web_app.py
```

---

## 📖 What is This?

AnCore Web App is a **user-friendly interface** for detecting fake news in Filipino articles. No technical knowledge needed!

### ✨ Features

- 🎯 **Simple Interface** - Just paste and click
- 📊 **Visual Results** - Color-coded predictions
- 💯 **Credibility Score** - Easy to understand 0-100 rating
- 📜 **History Tracking** - See your previous checks
- 🎨 **Beautiful Design** - Professional and clean
- 🔒 **Private** - Everything runs on your computer

---

## 🎯 How to Use (3 Simple Steps)

### Step 1: Launch the App
- Double-click `launch_web_app.bat`
- OR run: `streamlit run web_app.py`

### Step 2: Enter Article
- Paste a Filipino news article in the text box
- OR select a sample article to try

### Step 3: Get Results
- Click "🔍 Analyze Article"
- See if it's Real or Fake
- Check the confidence score

**That's it!** 🎉

---

## 📸 Screenshots

### Main Interface
```
┌─────────────────────────────────────────┐
│  🔍 AnCore - Fake News Detector        │
│  ────────────────────────────────────  │
│                                         │
│  📝 Enter News Article                 │
│  ┌─────────────────────────────────┐  │
│  │ Paste your article here...      │  │
│  │                                 │  │
│  │                                 │  │
│  └─────────────────────────────────┘  │
│                                         │
│        [🔍 Analyze Article]            │
│                                         │
└─────────────────────────────────────────┘
```

### Results Display
```
┌─────────────────────────────────────────┐
│  ✅ Real News                           │
│  Confidence: 92.3% (High)               │
│                                         │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐│
│  │Credibility │ │ Real     │ │ Fake    ││
│  │  92.3/100 │ │  92.3%   │ │  7.7%   ││
│  └──────────┘ └──────────┘ └─────────┘│
│                                         │
│  💡 High Confidence Prediction          │
│  The model is very confident about      │
│  this being real news.                  │
└─────────────────────────────────────────┘
```

---

## 🎨 Understanding the Results

### Color Codes

| Color | Meaning | Example |
|-------|---------|---------|
| 🟢 **Green** | Real News | Trusted, credible article |
| 🔴 **Red** | Fake News | Potentially false information |
| 🟡 **Yellow** | Uncertain | Needs more verification |

### Confidence Levels

| Level | Confidence | What It Means |
|-------|------------|---------------|
| 😊 **High** | 80-100% | Very reliable prediction |
| 🙂 **Medium** | 50-80% | Somewhat reliable, verify |
| 😐 **Low** | 0-50% | Uncertain, definitely verify |

### Credibility Score

```
0-30   ⚠️  Very Suspicious
31-50  ⚠️  Suspicious
51-70  🟡 Questionable
71-85  ✅ Likely Real
86-100 ✅ Very Likely Real
```

---

## ⚡ Quick Tips

### ✅ For Best Results:

1. **Use Complete Articles**
   - Paste the full article, not just headlines
   - Include at least 20 words

2. **Check the Confidence**
   - High confidence = More reliable
   - Low confidence = Verify elsewhere

3. **Use Multiple Sources**
   - Don't rely on AI alone
   - Cross-check important news

4. **Look at the Score**
   - Higher score = More credible
   - Lower score = More suspicious

### ⚠️ Important Reminders:

- This is **AI assistance**, not absolute truth
- **Always verify** important news from official sources
- Works best with **Filipino/Tagalog** text
- Requires at least **20 words** for accuracy

---

## 🔧 Troubleshooting

### Problem: Can't Launch the App

**Solution 1**: Install Streamlit
```powershell
pip install streamlit
```

**Solution 2**: Use the launcher
```
Double-click: launch_web_app.bat
```

---

### Problem: Browser Doesn't Open

**Solution**: Manually open your browser and go to:
```
http://localhost:8501
```

---

### Problem: Model Not Found

**Error Message**: "Model file not found"

**Solution**: Train the model first:
```powershell
python ancore_main.py --mode train
```

This will take 1-2 hours but only needs to be done once.

---

### Problem: App is Slow

**Solution**:
- First analysis is always slower (loading model)
- Next analyses are much faster
- Close other programs to free memory
- Restart the app if needed

---

## 📱 Using on Other Devices

### On Same WiFi Network

1. **On your computer**, find your IP address:
   ```powershell
   ipconfig
   ```
   Look for "IPv4 Address" (e.g., 192.168.1.100)

2. **On phone/tablet**, open browser:
   ```
   http://192.168.1.100:8501
   ```

3. **Use the app** from any device!

---

## 🎓 Example Usage

### Example 1: Checking Facebook Post

```
1. See a news post on Facebook
2. Copy the article text
3. Paste into AnCore Web App
4. Click "Analyze Article"
5. Check if it's Real or Fake
6. Verify from official sources if suspicious
```

### Example 2: Student Research

```
1. Find news articles for research
2. Check each article in AnCore
3. Use only articles with high credibility (≥80%)
4. Keep track using the History feature
5. Cite properly in your paper
```

### Example 3: Family Group Chat

```
1. Someone shares news in group chat
2. Copy the article
3. Check in AnCore
4. If marked Fake with high confidence:
   → Inform the group politely
   → Share the credibility score
   → Suggest official sources
```

---

## 📊 Features Overview

### Main Features

✅ **Simple Text Input**
- Large text area
- Character counter
- Word counter
- Sample articles included

✅ **Clear Results**
- Color-coded predictions
- Confidence percentage
- Credibility score (0-100)
- Visual progress bars

✅ **History Tracking**
- See all previous analyses
- Timestamps for each check
- Quick preview of articles
- Clear history option

✅ **Helpful Information**
- Model accuracy stats
- How to interpret results
- Tips for verification
- Important warnings

### Design Features

🎨 **Beautiful Interface**
- Professional design
- Easy to read
- Mobile-friendly
- Intuitive layout

🚀 **Fast Performance**
- Results in 2-5 seconds
- Cached model loading
- Smooth experience

🔒 **Privacy First**
- No data collection
- Local processing only
- Anonymous usage
- No login required

---

## 📞 Need Help?

### Common Questions

**Q: Is this 100% accurate?**
A: No. It's about 85-90% accurate. Always verify important news.

**Q: Can it detect all types of fake news?**
A: It's very good but not perfect. Use it as a helpful tool.

**Q: Does it work offline?**
A: Yes! After loading, it works without internet.

**Q: Is my data safe?**
A: Yes! Everything stays on your computer.

**Q: Can I use it on my phone?**
A: Yes! Connect to the same WiFi and use the network URL.

---

## 🎉 You're Ready!

**Start fighting fake news today!**

1. Launch the app: `launch_web_app.bat`
2. Paste an article
3. Click Analyze
4. Get results in seconds!

**Remember**: This is a tool to **help** you, not replace your judgment. Always think critically and verify important information! 🧠✨

---

## 📚 Additional Resources

- **Full Guide**: See `WEB_APP_GUIDE.md`
- **Technical Docs**: See `README.md`
- **Troubleshooting**: See `TROUBLESHOOTING.md`
- **Quick Start**: See `QUICKSTART.md`

---

**Happy fact-checking! 🔍✨**
