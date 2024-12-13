import streamlit as st
import pandas as pd
import numpy as np
from movieloader import MovieLensLoader
from interactive import InteractiveRecommender

# Cache the data loading
@st.cache_resource
def load_data():
    """Load data with caching and error handling"""
    try:
        loader = MovieLensLoader()
        
        with st.spinner('Loading movies data...'):
            movies_df = loader.load_movies()
        
        with st.spinner('Loading ratings data...'):
            ratings_df = loader.load_ratings()
            
        return movies_df, ratings_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

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
    with st.spinner('Initializing recommendation system...'):
        try:
            movies_df, ratings_df = load_data()
            st.session_state.recommender = InteractiveRecommender(movies_df, ratings_df)
            st.session_state.all_genres = set()
            for genres in movies_df['genres']:
                st.session_state.all_genres.update(genres)
            st.success("Successfully loaded movie database!")
        except Exception as e:
            st.error(f"Error initializing recommender: {str(e)}")
            st.stop()

# App title and description
st.title("🎬 Movie Recommendation System")
st.write("Get personalized movie recommendations based on your preferences!")

# Main layout
col1, col2 = st.columns([1, 3])

# Sidebar for user preferences
with col1:
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
    
    if st.button("Update Preferences", type="primary", use_container_width=True):
        st.session_state.user_preferences['genres'] = genre_preferences
        st.success("Preferences updated!")

# Main content area
with col2:
    if not st.session_state.user_preferences:
        st.info("👈 Please rate your favorite genres and click 'Update Preferences' to get started!")
    else:
        try:
            # Get recommendations
            recommender = st.session_state.recommender
            recommender.user_preferences = st.session_state.user_preferences
            recommender.user_ratings = st.session_state.user_ratings
            
            st.header("Your Personalized Movie Recommendations")
            
            with st.spinner("Generating recommendations..."):
                recommendations = recommender.get_recommendations(n_recommendations=10)
            
            # Display recommendations in a grid
            for idx, movie in enumerate(recommendations.iterrows(), 1):
                movie = movie[1]  # Get the row data
                with st.expander(f"{idx}. {movie['title']}", expanded=True):
                    col_info, col_rating = st.columns([2, 1])
                    
                    with col_info:
                        st.markdown(f"**Genres:** {', '.join(movie['genres'])}")
                        st.markdown(f"**Average Rating:** ⭐ {movie['rating_mean']:.1f}/5")
                        st.markdown(f"**Number of Ratings:** 🎬 {int(movie['rating_count']):,}")
                        st.markdown(f"**Match Score:** 🎯 {movie['score']:.2f}/5")
                    
                    with col_rating:
                        # Add rating input for each movie
                        rating = st.slider(
                            "Rate this movie",
                            min_value=0.0,
                            max_value=5.0,
                            value=0.0,
                            step=0.5,
                            key=f"rating_{movie['movieId']}",
                            help="Rate this movie to improve future recommendations"
                        )
                        
                        if rating > 0:
                            st.session_state.user_ratings[movie['movieId']] = rating
            
            # Add a refresh button
            if st.button("Get New Recommendations", type="secondary", use_container_width=True):
                st.rerun()
                
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
            
# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit and MovieLens dataset</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Add retry button for data loading if needed
if st.session_state.get('data_load_failed', False):
    if st.button("Retry Loading Data"):
        st.session_state.clear()
        st.rerun()
