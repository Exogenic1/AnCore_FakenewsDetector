"""
AnCore Web Application
User-friendly web interface for fake news detection using the trained mBERT model
"""

import streamlit as st
import torch
import os
from datetime import datetime
import pandas as pd
from transformers import AutoTokenizer

from ancore_config import Config
from ancore_model import mBERTClassifier, CredibilityAssessor

# Page configuration
st.set_page_config(
    page_title="AnCore - Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 15px;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .real-news {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .fake-news {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .neutral {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        padding: 20px 0;
    }
    h2 {
        color: #34495e;
    }
    .info-box {
        background-color: #e7f3ff;
        border-left: 5px solid #2196F3;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model (cached for performance)"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = mBERTClassifier(num_labels=Config.NUM_LABELS)
        
        model_path = os.path.join(Config.MODEL_SAVE_DIR, 'best_model.pt')
        
        if not os.path.exists(model_path):
            return None, None, "Model file not found"
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        
        return model, tokenizer, device
    except Exception as e:
        return None, None, str(e)


def predict_article(text, model, tokenizer, device):
    """Predict if an article is fake or real"""
    try:
        # Tokenize the text
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=Config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Create credibility assessor
        assessor = CredibilityAssessor(model, device)
        
        # Get prediction
        results = assessor.assess_credibility(input_ids, attention_mask)
        
        return results[0]
    except Exception as e:
        return {"error": str(e)}


def display_result(result):
    """Display the prediction result with beautiful formatting"""
    if "error" in result:
        st.error(f"Error: {result['error']}")
        return
    
    prediction = result['prediction']
    confidence = result['confidence']
    prob_real = result['probability_real']
    prob_fake = result['probability_fake']
    confidence_level = result['confidence_level']
    
    # Determine styling based on prediction
    if prediction == "Real News":
        card_class = "real-news"
        emoji = "✅"
        color = "#28a745"
    else:
        card_class = "fake-news"
        emoji = "⚠️"
        color = "#dc3545"
    
    # Display result card
    st.markdown(f"""
        <div class="{card_class}">
            <h2 style="color: {color}; margin: 0;">{emoji} {prediction}</h2>
            <p style="font-size: 18px; margin: 10px 0;">
                Confidence: <strong>{confidence:.1%}</strong> ({confidence_level})
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Display detailed metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #2c3e50; margin: 0;">Credibility Score</h3>
                <p style="font-size: 32px; font-weight: bold; color: {color}; margin: 10px 0;">
                    {prob_real * 100:.1f}/100
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #2c3e50; margin: 0;">Real News</h3>
                <p style="font-size: 32px; font-weight: bold; color: #28a745; margin: 10px 0;">
                    {prob_real:.1%}
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #2c3e50; margin: 0;">Fake News</h3>
                <p style="font-size: 32px; font-weight: bold; color: #dc3545; margin: 10px 0;">
                    {prob_fake:.1%}
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Progress bar for visualization
    st.markdown("### 📊 Probability Distribution")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Real News Probability", f"{prob_real:.2%}")
        st.progress(prob_real)
    
    with col2:
        st.metric("Fake News Probability", f"{prob_fake:.2%}")
        st.progress(prob_fake)
    
    # Interpretation guide
    st.markdown("---")
    st.markdown("### 💡 How to Interpret the Results")
    
    if confidence >= Config.HIGH_CONFIDENCE_THRESHOLD:
        st.success(f"""
        **High Confidence Prediction** ({confidence:.1%})
        
        The model is very confident about this prediction. The article characteristics strongly 
        match those of {prediction.lower()}.
        """)
    elif confidence >= Config.LOW_CONFIDENCE_THRESHOLD:
        st.info(f"""
        **Medium Confidence Prediction** ({confidence:.1%})
        
        The model has moderate confidence in this prediction. The article shows some 
        characteristics of {prediction.lower()}, but further verification is recommended.
        """)
    else:
        st.warning(f"""
        **Low Confidence Prediction** ({confidence:.1%})
        
        The model is uncertain about this prediction. The article shows mixed characteristics. 
        Please verify through other sources.
        """)


def main():
    """Main application"""
    
    # Header
    st.markdown("""
        <h1>🔍 AnCore - Fake News Detector</h1>
        <p style="text-align: center; font-size: 18px; color: #7f8c8d;">
            AI-Powered News Credibility Assessment for Filipino Articles
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load model
    with st.spinner("🤖 Loading AI model... Please wait..."):
        model, tokenizer, device = load_model()
    
    if model is None:
        st.error(f"""
        ❌ **Model Loading Failed**
        
        {device}
        
        Please ensure:
        1. The model file exists at `output/models/best_model.pt`
        2. You have trained the model using: `python ancore_main.py --mode train`
        """)
        st.stop()
    
    st.success(f"✅ Model loaded successfully! Running on: **{str(device).upper()}**")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📖 About AnCore")
        st.info("""
        **AnCore** uses state-of-the-art AI (mBERT) to analyze Filipino news articles 
        and determine their credibility.
        
        The system was trained on 3,206 expertly-labeled articles to distinguish 
        between real and fake news.
        """)
        
        st.markdown("### 🎯 How to Use")
        st.markdown("""
        1. **Paste** or **type** a Filipino news article in the text box
        2. Click **"Analyze Article"** button
        3. Review the results and credibility score
        4. Check the confidence level
        """)
        
        st.markdown("### ⚙️ Model Information")
        st.markdown(f"""
        - **Model**: mBERT (Multilingual BERT)
        - **Parameters**: 177M
        - **Training Data**: 3,206 articles
        - **Languages**: Filipino/Tagalog
        """)
        
        st.markdown("### ⚠️ Important Notes")
        st.warning("""
        - This is an AI prediction, not a definitive fact
        - Always verify important news from multiple sources
        - The model works best with complete articles
        - Short texts may have lower accuracy
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Quick Statistics")
        st.metric("Model Accuracy", "~85-90%")
        st.metric("Processing Time", "<5 seconds")
    
    # Main content area
    st.markdown("## 📝 Enter News Article")
    
    # Sample articles for testing
    sample_articles = {
        "Sample Real News": """Ayon sa TheWrap.com, naghain ng kaso si Krupa, 35, noong Huwebes dahil nakaranas umano siya ng emotional distress bunga ng mga malisyosong pahayag ni Glanville, hindi lamang tungkol sa maselang bahagi ng kanyang katawan kundi pati na rin sa kanyang buhay pag-ibig.""",
        
        "Sample Fake News": """Isiniwalat ni DZME Radio host na si Mark Lopez sa kanyang social media account ang natukalasan niya tungkol sa taktika na ginagamit umano ng kampo ni Liberal Party senatoriable Mar Roxas para makabalik sa senado. Base sa mga impormasyon at screenshots na pinost ni Lopez sa kanyang Facebook account, tila gumagamit daw ang kampo ni Roxas ng mga taong may maraming followings sa social media para magmulat ng maling impormasyon laban sa mga kalaban nito sa pulitika. Ayon pa sa post, binabayaran umano ang mga influencers upang magkalat ng fake news at propaganda.""",
        
        "Type Your Own": ""
    }
    
    # Sample selection
    sample_choice = st.selectbox(
        "Choose a sample article or select 'Type Your Own':",
        options=list(sample_articles.keys()),
        index=2
    )
    
    # Text input
    if sample_choice == "Type Your Own":
        article_text = st.text_area(
            "Paste or type the news article here:",
            height=300,
            placeholder="Enter a Filipino news article here...\n\nExample: Ayon sa ulat, nangyari ang insidente noong...",
            help="Paste the complete article for best results. The model analyzes the writing style, content, and patterns."
        )
    else:
        article_text = st.text_area(
            "Article text (you can edit this):",
            value=sample_articles[sample_choice],
            height=300,
            help="This is a sample article. You can edit it or replace it with your own."
        )
    
    # Character count
    char_count = len(article_text)
    word_count = len(article_text.split())
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Characters", char_count)
    with col2:
        st.metric("Words", word_count)
    with col3:
        if word_count < 20:
            st.metric("Status", "Too Short ⚠️")
        else:
            st.metric("Status", "Ready ✅")
    
    # Warning for short texts
    if article_text and word_count < 20:
        st.warning("⚠️ **Article is very short!** For better accuracy, please provide at least 20 words.")
    
    # Analyze button
    st.markdown("###")
    analyze_button = st.button("🔍 Analyze Article", disabled=not article_text.strip())
    
    # Analysis section
    if analyze_button:
        if not article_text.strip():
            st.error("❌ Please enter some text to analyze.")
        else:
            with st.spinner("🤔 Analyzing article... This may take a few seconds..."):
                result = predict_article(article_text, model, tokenizer, device)
            
            st.markdown("---")
            st.markdown("## 📊 Analysis Results")
            display_result(result)
            
            # Save to history (optional feature)
            if 'history' not in st.session_state:
                st.session_state.history = []
            
            st.session_state.history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'prediction': result.get('prediction', 'Unknown'),
                'confidence': result.get('confidence', 0),
                'text_preview': article_text[:100] + "..."
            })
    
    # History section
    if 'history' in st.session_state and st.session_state.history:
        st.markdown("---")
        st.markdown("## 📜 Analysis History")
        
        with st.expander("View Previous Analyses", expanded=False):
            history_df = pd.DataFrame(st.session_state.history)
            
            # Display in reverse order (newest first)
            for idx, row in history_df.iloc[::-1].iterrows():
                prediction = row['prediction']
                emoji = "✅" if prediction == "Real News" else "⚠️"
                
                st.markdown(f"""
                **{emoji} {prediction}** - {row['confidence']:.1%} confidence  
                *{row['timestamp']}*  
                _{row['text_preview']}_
                """)
                st.markdown("---")
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.history = []
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #7f8c8d; padding: 20px;">
            <p><strong>AnCore - News Article Credibility Assessment</strong></p>
            <p>Powered by mBERT (Multilingual BERT) | Trained on Filipino News Dataset</p>
            <p style="font-size: 12px;">⚠️ This is an AI-powered tool. Always verify important news from multiple trusted sources.</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
