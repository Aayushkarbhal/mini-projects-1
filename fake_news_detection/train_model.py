import pandas as pd
import re
import string
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

train_df = pd.read_csv("fake_news_train.csv")
test_df = pd.read_csv("fake_news_test.csv")

print("Train Dataset Loaded Successfully!")
print(train_df.head())
print("\nColumns:", train_df.columns)

if 'title' in train_df.columns and 'text' in train_df.columns:
    train_df['content'] = train_df['title'].fillna('') + " " + train_df['text'].fillna('')
    test_df['content'] = test_df['title'].fillna('') + " " + test_df['text'].fillna('')
elif 'text' in train_df.columns:
    train_df['content'] = train_df['text'].fillna('')
    test_df['content'] = test_df['text'].fillna('')
else:
    raise ValueError("Dataset must contain 'text' or 'title' columns.")

if 'label' not in train_df.columns:
    if 'class' in train_df.columns:
        train_df['label'] = train_df['class']
        test_df['label'] = test_df['class']
    elif 'target' in train_df.columns:
        train_df['label'] = train_df['target']
        test_df['label'] = test_df['target']
    else:
        raise ValueError("Dataset must contain a 'label', 'class', or 'target' column.")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

train_df['content'] = train_df['content'].apply(clean_text)
test_df['content'] = test_df['content'].apply(clean_text)

train_df['label'] = train_df['label'].map({'Fake': 0, 'Real': 1})
test_df['label'] = test_df['label'].map({'Fake': 0, 'Real': 1})

X_train = train_df['content']
y_train = train_df['label']

X_test = test_df['content']
y_test = test_df['label']

print("\nUnique train labels after mapping:", y_train.unique())
print("Unique test labels after mapping:", y_test.unique())

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")