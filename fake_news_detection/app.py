import streamlit as st
import joblib
import re
import string

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }
    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #00C2FF;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #B0B0B0;
        margin-bottom: 30px;
    }
    .info-box {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 20px;
        border: 1px solid #333;
    }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

st.markdown('<div class="title">📰 Fake News Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Check whether a news article is Fake or Real using Machine Learning</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("📌 Project Info")
    st.write("**Project Name:** Fake News Detection")
    st.write("**Model Used:** Logistic Regression")
    st.write("**Technique:** TF-IDF Vectorization")
    st.write("**Built With:** Python, Scikit-learn, Streamlit")
    st.write("---")
    st.info("Tip: Paste full news-like text for better predictions.")

st.markdown("""
<div class="info-box">
This project classifies news articles as <b>Fake</b> or <b>Real</b> using Natural Language Processing (NLP) and Machine Learning.
</div>
""", unsafe_allow_html=True)

user_input = st.text_area("✍️ Enter News Text Here", height=220, placeholder="Paste a news article or headline here...")

if st.button("🔍 Analyze News"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some news text before prediction.")
    else:
        cleaned = clean_text(user_input)
        transformed = vectorizer.transform([cleaned])

        prediction = model.predict(transformed)[0]
        confidence = model.predict_proba(transformed)[0]

        real_conf = confidence[1] * 100
        fake_conf = confidence[0] * 100

        st.subheader("📊 Prediction Result")

        if prediction == 1:
            st.success("✅ This news appears to be REAL.")
        else:
            st.error("🚨 This news appears to be FAKE.")

        st.progress(int(max(real_conf, fake_conf)))

        col1, col2 = st.columns(2)
        with col1:
            st.metric("🟢 Real Confidence", f"{real_conf:.2f}%")
        with col2:
            st.metric("🔴 Fake Confidence", f"{fake_conf:.2f}%")

st.write("---")
st.caption("Made by a fresher learning AI/ML • Fake News Detection Project")