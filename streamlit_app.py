import streamlit as st
import pandas as pd
import numpy as np
from movieloader import MovieLensLoader
from interactive import InteractiveRecommender

# Set page config
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# Initialize session state
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {}
if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = {}
if 'recommender' not in st.session_state:
    # Load data
    data_path = 'ml-25m'  # This will need to be updated for deployment
    loader = MovieLensLoader(data_path)
    
    movies_df = loader.load_movies()
    ratings_df = loader.load_ratings()
    
    st.session_state.recommender = InteractiveRecommender(movies_df, ratings_df)
    st.session_state.all_genres = set()
    for genres in movies_df['genres']:
        st.session_state.all_genres.update(genres)

# App title and description
st.title("🎬 Movie Recommendation System")
st.write("Get personalized movie recommendations based on your preferences!")

# Sidebar for user preferences
with st.sidebar:
    st.header("Your Preferences")
    st.subheader("Rate Your Favorite Genres")
    
    genre_preferences = {}
    for genre in sorted(st.session_state.all_genres):
        genre_preferences[genre] = st.slider(
            f"{genre}",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.5,
            key=f"genre_{genre}"
        )
    
    if st.button("Update Preferences"):
        st.session_state.user_preferences['genres'] = genre_preferences
        st.success("Preferences updated!")

# Main content
if not st.session_state.user_preferences:
    st.info("👈 Please rate your favorite genres in the sidebar to get started!")
else:
    # Get recommendations
    recommender = st.session_state.recommender
    recommender.user_preferences = st.session_state.user_preferences
    recommender.user_ratings = st.session_state.user_ratings
    
    recommendations = recommender.get_recommendations(n_recommendations=10)
    
    st.header("Your Personalized Movie Recommendations")
    
    # Display recommendations in a grid
    cols = st.columns(2)
    for idx, movie in enumerate(recommendations.iterrows()):
        movie = movie[1]  # Get the row data
        with cols[idx % 2]:
            with st.expander(f"{movie['title']}", expanded=True):
                st.write(f"**Genres:** {', '.join(movie['genres'])}")
                st.write(f"**Average Rating:** ⭐ {movie['rating_mean']:.1f}/5")
                st.write(f"**Number of Ratings:** 🎬 {int(movie['rating_count']):,}")
                st.write(f"**Match Score:** 🎯 {movie['score']:.2f}/5")
                
                # Add rating input for each movie
                rating = st.slider(
                    "Rate this movie",
                    min_value=0.0,
                    max_value=5.0,
                    value=0.0,
                    step=0.5,
                    key=f"rating_{movie['movieId']}"
                )
                
                if rating > 0:
                    st.session_state.user_ratings[movie['movieId']] = rating
    
    # Add a refresh button
    if st.button("Get New Recommendations"):
        st.experimental_rerun()
