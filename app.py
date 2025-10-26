import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load preprocessed files
movies = pd.read_csv("movies_preprocessed.csv")

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# recompute tfidf matrix
tfidf_matrix = tfidf.transform(movies["combined"])
content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

st.title("🎬 Simple Movie Recommender (Hybrid)")
movie_input = st.text_input("Enter movie title (partial allowed):", "Toy Story")
content_w = st.slider("Content weight", 0.0, 1.0, 0.6)

if st.button("Recommend"):
    mask = movies["title"].str.contains(movie_input, case=False, na=False)
    if mask.sum() == 0:
        st.error("No movie found.")
    else:
        movie_idx = movies[mask].iloc[0].name
        content_scores = content_sim[movie_idx]
        hybrid_scores = content_w * content_scores
        top_idx = np.argsort(hybrid_scores)[::-1]
        top_idx = [i for i in top_idx if i != movie_idx][:5]
        for i in top_idx:
            st.write(f"**{movies.loc[i,'title']}** — {movies.loc[i,'genres']}")
