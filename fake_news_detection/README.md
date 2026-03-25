# Fake News Detection Using Machine Learning

## Project Overview

This is a beginner-level machine learning project made to classify whether a news article is **Fake** or **Real** using Natural Language Processing (NLP).

I made this project as part of my AI/ML learning journey. As a fresher, my main goal was to understand how text data can be processed and used in machine learning for real-world applications.

This project took me around **1 week** to complete, including:

- understanding the dataset
- learning basic NLP preprocessing
- training the model
- building a simple user interface using Streamlit

---

## Objective

The main objective of this project is to detect fake news articles based on their textual content using a machine learning model.

Fake news has become a major issue in today’s digital world, and this project helped me understand how AI can be used to solve such practical problems.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- NLP (Text Cleaning + TF-IDF)

---

## What I Learned From This Project

While building this project, I learned:

- how to work with real-world CSV datasets
- how to clean and preprocess text data
- how TF-IDF converts text into numerical form
- how Logistic Regression works for classification
- how to save and load trained models
- how to create a simple frontend using Streamlit

This project also helped me improve my confidence in building end-to-end ML projects as a beginner.

---

## How the Project Works

1. The fake and real news datasets are loaded.
2. The title and text columns are combined.
3. The text is cleaned by removing punctuation, URLs, and unnecessary characters.
4. The cleaned text is converted into numerical vectors using **TF-IDF Vectorization**.
5. A **Logistic Regression** model is trained on the dataset.
6. The model predicts whether the input news is fake or real.
7. A simple Streamlit app is used to test the prediction interactively.

---

## Project Structure

```bash
fake_news_detection/
│── app.py
│── train_model.py
│── fake_news_train.csv
│── fake_news_test.csv
│── fake_news_model.pkl
│── vectorizer.pkl
│── requirements.txt
│── README.md
│── screenshots/
```
