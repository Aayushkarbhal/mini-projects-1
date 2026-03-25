import os
import PyPDF2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# FUNCTION TO EXTRACT TEXT FROM PDF
# -----------------------------
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
    except:
        text = ""
    return text

# -----------------------------
# FUNCTION TO RANK RESUMES
# -----------------------------
def rank_resumes(uploaded_files, job_description):
    resume_data = []

    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resume_data.append({
            "Resume Name": file.name,
            "Resume Text": text
        })

    df = pd.DataFrame(resume_data)

    # Combine JD with resume texts
    documents = [job_description] + df["Resume Text"].tolist()

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Cosine Similarity
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    df["Match Score (%)"] = (similarity_scores * 100).round(2)

    # Sort by best match
    df = df.sort_values(by="Match Score (%)", ascending=False).reset_index(drop=True)

    return df[["Resume Name", "Match Score (%)"]]