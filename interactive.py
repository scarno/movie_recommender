import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random

class InteractiveRecommender:
    def __init__(self, movies_df, ratings_df, min_ratings=1000):
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.user_preferences = {}
        self.user_ratings = {}
        self.min_ratings = min_ratings
        
        # Calculate movie stats once
        self.movie_stats = self.calculate_movie_stats()
        
    def calculate_movie_stats(self):
        """Calculate popularity metrics for all movies"""
        stats = self.ratings_df.groupby('movieId').agg({
            'rating': ['count', 'mean']
        }).reset_index()
        stats.columns = ['movieId', 'rating_count', 'rating_mean']
        
        # Filter out movies with too few ratings
        stats = stats[stats['rating_count'] >= self.min_ratings]
        
        # Calculate popularity score
        stats['popularity_score'] = (stats['rating_count'] - stats['rating_count'].min()) / \
                                  (stats['rating_count'].max() - stats['rating_count'].min())
        
        return stats
    
    def get_popular_by_genre(self, genre, n=5):
        """Get popular movies from a specific genre"""
        genre_movies = self.movies_df[self.movies_df['genres'].apply(lambda x: genre in x)]
        genre_movies = genre_movies.merge(self.movie_stats, on='movieId')
        genre_movies = genre_movies.sort_values(['rating_count', 'rating_mean'], 
                                              ascending=[False, False])
        return genre_movies.head(n)
    
    def initialize_user_preferences(self):
        """Interactive questionnaire to initialize user preferences"""
        print("\nWelcome to the Movie Recommender System!")
        print("\nLet's get to know your movie preferences...")
        
        # Get genre preferences
        print("\nFirst, let's talk about movie genres.")
        all_genres = set()
        for genres in self.movies_df['genres']:
            all_genres.update(genres)
        
        print("\nPlease rate how much you enjoy the following genres from 1-5 (5 being love it):")
        genre_preferences = {}
        for genre in sorted(all_genres):
            while True:
                try:
                    rating = float(input(f"{genre}: "))
                    if 1 <= rating <= 5:
                        genre_preferences[genre] = rating
                        break
                    else:
                        print("Please enter a number between 1 and 5")
                except ValueError:
                    print("Please enter a valid number")
        
        self.user_preferences['genres'] = genre_preferences
        
        # Get ratings for popular movies
        print("\nNow, let's get your ratings for some popular movies.")
        print("Enter a rating 1-5, or 0 if you haven't seen it.\n")
        
        # Get movies from top genres
        top_genres = sorted(genre_preferences.items(), key=lambda x: x[1], reverse=True)[:3]
        for genre, _ in top_genres:
            popular_movies = self.get_popular_by_genre(genre)
            for _, movie in popular_movies.iterrows():
                while True:
                    try:
                        rating = float(input(f"{movie['title']} ({genre}) "
                                          f"[Avg Rating: {movie['rating_mean']:.1f}/5 from "
                                          f"{int(movie['rating_count']):,} ratings]: "))
                        if 0 <= rating <= 5:
                            if rating > 0:
                                self.user_ratings[movie['movieId']] = rating
                            break
                        else:
                            print("Please enter a number between 0 and 5")
                    except ValueError:
                        print("Please enter a valid number")
    
    def calculate_movie_scores(self):
        """Calculate scores for all movies based on preferences and popularity"""
        movies_with_stats = self.movies_df.merge(self.movie_stats, on='movieId', how='inner')
        
        scores = []
        for _, movie in movies_with_stats.iterrows():
            # Genre score (0-5 scale)
            genre_score = 0
            genre_weights = 0
            for genre in movie['genres']:
                if genre in self.user_preferences['genres']:
                    weight = self.user_preferences['genres'][genre]
                    genre_score += weight
                    genre_weights += 1
            
            if genre_weights > 0:
                genre_score = genre_score / genre_weights
            
            # Normalize scores
            popularity_score = movie['popularity_score']
            rating_score = (movie['rating_mean'] - 1) / 4  # Convert 1-5 scale to 0-1
            
            # Calculate final score (0-5 scale)
            final_score = (genre_score * 0.6 +                    # Genre preference (60%)
                         rating_score * 5 * 0.25 +                # Average rating (25%)
                         popularity_score * 5 * 0.15)             # Popularity (15%)
            
            scores.append(float(final_score))
        
        return np.array(scores)
    
    def get_recommendations(self, n_recommendations=5):
        """Generate diverse recommendations based on user preferences and movie popularity"""
        scores = self.calculate_movie_scores()
        
        # Create recommendations dataframe
        recommendations = self.movies_df.merge(self.movie_stats, on='movieId', how='inner')
        recommendations['score'] = scores
        
        # Filter out movies the user has rated
        rated_movies = set(self.user_ratings.keys())
        recommendations = recommendations[~recommendations['movieId'].isin(rated_movies)]
        
        # Sort by score
        recommendations = recommendations.sort_values('score', ascending=False)
        
        # Ensure genre diversity while maintaining high scores
        top_recommendations = []
        seen_genres = set()
        
        for _, movie in recommendations.iterrows():
            if len(top_recommendations) >= n_recommendations:
                break
            
            movie_genres = set(movie['genres'])
            primary_genres = {g for g in movie_genres 
                            if g in self.user_preferences 
                            and self.user_preferences[g] >= 4}
            
            if (not primary_genres.intersection(seen_genres) or 
                len(top_recommendations) < 2):
                top_recommendations.append(movie)
                seen_genres.update(primary_genres)
        
        return pd.DataFrame(top_recommendations)

def main():
    from movieloader import MovieLensLoader
    
    print("Loading movie database...")
    data_path = '/Users/danielscarnavack/MovieRec/ml-25m'
    loader = MovieLensLoader(data_path)
    
    movies_df = loader.load_movies()
    ratings_df = loader.load_ratings()
    
    recommender = InteractiveRecommender(movies_df, ratings_df)
    recommender.initialize_user_preferences()
    
    print("\nBased on your preferences, here are some movies you might enjoy:")
    recommendations = recommender.get_recommendations()
    
    for i, movie in enumerate(recommendations.iterrows(), 1):
        movie = movie[1]  # Get the row data
        print(f"\n{i}. {movie['title']}")
        print(f"   Genres: {', '.join(movie['genres'])}")
        print(f"   Average Rating: {movie['rating_mean']:.1f}/5 from {int(movie['rating_count']):,} ratings")
        print(f"   Match Score: {movie['score']:.2f}/5")

if __name__ == "__main__":
    main()
