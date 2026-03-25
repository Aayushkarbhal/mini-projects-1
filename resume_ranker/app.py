import streamlit as st
from resume_ranker import rank_resumes

st.set_page_config(
    page_title="AI Resume Ranker",
    page_icon="📄",
    layout="centered"
)

st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 40px;
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
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">📄 AI-Powered Resume Ranker</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload resumes and rank them based on job description relevance</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("📌 Project Info")
    st.write("**Model Used:** TF-IDF + Cosine Similarity")
    st.write("**Built With:** Python, Scikit-learn, Streamlit")
    st.write("---")
    st.info("Tip: Upload 2 or more resumes for best results.")

st.markdown("""
<div class="info-box">
This project ranks resumes based on how closely they match a given job description using Natural Language Processing (NLP).
</div>
""", unsafe_allow_html=True)

job_description = st.text_area("📝 Enter Job Description", height=180)

uploaded_files = st.file_uploader(
    "📂 Upload Resume PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("🚀 Rank Resumes"):
    if not job_description.strip():
        st.warning("⚠️ Please enter a job description.")
    elif not uploaded_files:
        st.warning("⚠️ Please upload at least one resume.")
    else:
        results = rank_resumes(uploaded_files, job_description)

        st.subheader("📊 Resume Ranking Results")
        st.dataframe(results, use_container_width=True)

        top_resume = results.iloc[0]
        st.success(f"🏆 Best Match: {top_resume['Resume Name']} with {top_resume['Match Score (%)']}% match")
        
st.write("---")
st.caption("Made by a fresher learning AI/ML • Resume Ranker Project")