import streamlit as st
from recommender import MovieRecommender

# Load the recommender
@st.cache_resource
def load_recommender():
    return MovieRecommender("data/movie_dataset.csv")

recommender = load_recommender()

# UI
st.title("🎥 Movie Recommendation System")

# Get movie list from the recommender's dataset
movie_titles = sorted(recommender.df['title'].unique())
selected_movie = st.selectbox("Choose a movie you like", movie_titles)
if st.button("Get Recommendations"):
    recommendations = recommender.recommend(selected_movie)

    st.subheader(f"Top Recommendations based on '{selected_movie}':")
    if isinstance(recommendations, list):
        for idx, title in enumerate(recommendations, 1):
            st.write(f"{idx}. {title}")
    else:
        st.warning(recommendations)
