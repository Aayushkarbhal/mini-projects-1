# Movie Recommendation System Using Machine Learning

## Project Overview

This is a beginner-level AI/ML project that recommends movies based on user preferences. The system suggests similar movies using content-based filtering techniques.

I built this project as part of my AI/ML learning journey. As a fresher, I wanted to understand how recommendation systems work in platforms like Netflix or Amazon Prime. This project took me around **1 week** to complete while learning the concepts step by step.

---

## Objective

The main objective of this project is to recommend movies based on similarity in genre, description, and other features using machine learning techniques.

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Natural Language Processing (TF-IDF + Cosine Similarity)

---

## What I Learned

While building this project, I learned:

- how recommendation systems work
- how to use real-world datasets from Kaggle
- how to clean and preprocess text data
- how TF-IDF converts text into numerical form
- how cosine similarity is used to find similar items
- how to build a user-friendly interface using Streamlit

---

## Features

- Movie selection using dropdown
- Content-based recommendation system
- Genre-based filtering
- Displays movie ratings
- Clean and user-friendly UI

---

## How the Project Works

1. The movie dataset is loaded from a CSV file.
2. Important features like genre and description are combined.
3. Text data is cleaned and processed.
4. TF-IDF vectorization is applied to convert text into numerical form.
5. Cosine similarity is used to compare movies.
6. The system recommends top similar movies based on the selected movie.
7. Results are displayed in a Streamlit web application.

---

## Project Structure

```bash
movie_recommender/
│── app.py
│── movie_recommender.py
│── movies.csv
│── requirements.txt
│── README.md
│── screenshots/
```
