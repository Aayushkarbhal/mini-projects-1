# AI-Powered Resume Ranker Using NLP

## Project Overview

This is a beginner-level AI/ML project that ranks resumes based on how well they match a given job description.

I made this project as part of my learning journey in Artificial Intelligence and Machine Learning. As a fresher, I wanted to understand how Natural Language Processing (NLP) can be used in real-world applications like recruitment and resume screening.

This project took me around **1 week** to complete while learning concepts step by step.

---

## Objective

The main objective of this project is to automate the process of ranking resumes according to a job description using text similarity techniques.

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- PyPDF2
- NLP (TF-IDF + Cosine Similarity)

---

## What I Learned

- how to extract text from PDF resumes
- how TF-IDF works for text representation
- how cosine similarity helps compare text relevance
- how to build a simple interactive UI using Streamlit

---

## How the Project Works

1. The user enters a job description.
2. The user uploads multiple resume PDFs.
3. The system extracts text from each resume.
4. TF-IDF vectorization is applied to convert text into numerical form.
5. Cosine similarity is used to compare each resume with the job description.
6. Resumes are ranked based on match score.

---

## Project Structure

```bash
resume_ranker/
│── app.py
│── resume_ranker.py
│── requirements.txt
│── README.md
│── jd.txt
│── sample_resumes/
│── screenshots/
```
