import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("movies.csv")
df = df.fillna("")

# Column names
TITLE_COL = "title"
GENRE_COL = "genres"
DESC_COL = "overview" if "overview" in df.columns else "description"
RATING_COL = "vote_average" if "vote_average" in df.columns else "rating"

# -----------------------------
# CLEAN GENRES FUNCTION
# -----------------------------
def clean_genres(genre_value):
    if pd.isna(genre_value) or genre_value == "":
        return "Unknown"

    genre_str = str(genre_value)

    # If format is like "Action|Adventure|Sci-Fi"
    if "|" in genre_str:
        return " • ".join([g.strip() for g in genre_str.split("|") if g.strip()])

    # If format is like "['Action', 'Adventure']"
    try:
        parsed = ast.literal_eval(genre_str)
        if isinstance(parsed, list):
            cleaned = []
            for item in parsed:
                if isinstance(item, dict) and "name" in item:
                    cleaned.append(item["name"])
                else:
                    cleaned.append(str(item))
            return " • ".join(cleaned)
    except:
        pass

    # Fallback
    genre_str = genre_str.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    genre_str = genre_str.replace(",", " • ")
    return genre_str.strip()

# Clean genres
df["clean_genres"] = df[GENRE_COL].apply(clean_genres)

# Combine features
df["content"] = df["clean_genres"] + " " + df[DESC_COL].astype(str)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["content"])

# Similarity
similarity = cosine_similarity(tfidf_matrix)

# -----------------------------
# RECOMMEND FUNCTION
# -----------------------------
def recommend_movies(movie_title, num_recommendations=5):
    movie_title = movie_title.lower().strip()

    matches = df[df[TITLE_COL].str.lower() == movie_title]

    if matches.empty:
        return None

    idx = matches.index[0]

    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]

    results = []
    for i in sim_scores:
        movie = df.iloc[i[0]]
        results.append({
            "title": movie[TITLE_COL],
            "rating": movie.get(RATING_COL, "N/A"),
            "genre": movie.get("clean_genres", "Unknown")
        })

    return results

# -----------------------------
# GET MOVIE TITLES
# -----------------------------
def get_movie_titles():
    return df[TITLE_COL].dropna().tolist()

# -----------------------------
# GET UNIQUE GENRES
# -----------------------------
def get_genres():
    genre_set = set()
    for g in df["clean_genres"]:
        for item in str(g).split("•"):
            item = item.strip()
            if item:
                genre_set.add(item)
    return sorted(list(genre_set))