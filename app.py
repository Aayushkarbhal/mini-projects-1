import streamlit as st
from movie_recommender import recommend_movies, get_movie_titles, get_genres

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
    <style>
    .movie-card {
        background-color: #1E1E1E;
        padding: 18px;
        border-radius: 16px;
        margin-bottom: 15px;
        border: 1px solid #333;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    .movie-title {
        font-size: 24px;
        font-weight: bold;
        color: #FF4B4B;
    }
    .movie-meta {
        font-size: 16px;
        color: #D3D3D3;
        margin-top: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.title("🎬 Movie Recommendation System")
st.write("Get clean and personalized movie recommendations based on your selected movie.")

# Sidebar
with st.sidebar:
    st.header("🎯 Filters")
    selected_genre = st.selectbox("Select Genre", ["All"] + get_genres())
    num_recommendations = st.slider("Number of Recommendations", 3, 10, 5)

# Movie selection
movie_list = get_movie_titles()
selected_movie = st.selectbox("🎥 Choose a Movie", movie_list)

# Recommend
if st.button("🍿 Recommend"):
    results = recommend_movies(selected_movie, num_recommendations)

    if not results:
        st.error("Movie not found.")
    else:
        st.subheader("✨ Recommended Movies")

        shown = 0
        for movie in results:
            if selected_genre != "All" and selected_genre not in movie["genre"]:
                continue

            st.markdown(f"""
            <div class="movie-card">
                <div class="movie-title">🎞️ {movie['title']}</div>
                <div class="movie-meta">⭐ Rating: {movie['rating']}</div>
                <div class="movie-meta">🎭 Genre: {movie['genre']}</div>
            </div>
            """, unsafe_allow_html=True)

            shown += 1

        if shown == 0:
            st.warning("No recommendations matched the selected genre filter.")
