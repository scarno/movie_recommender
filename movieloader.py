import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix

class MovieLensLoader:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        
    def load_ratings(self):
        """Load and preprocess ratings data"""
        ratings_path = self.data_path / 'ratings.csv'
        ratings = pd.read_csv(ratings_path)
        ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
        return ratings
    
    def load_movies(self):
        """Load and preprocess movies data"""
        movies_path = self.data_path / 'movies.csv'
        movies = pd.read_csv(movies_path)
        movies['genres'] = movies['genres'].str.split('|')
        return movies
    
    def load_tags(self):
        """Load and preprocess tags data"""
        tags_path = self.data_path / 'tags.csv'
        tags = pd.read_csv(tags_path)
        tags['timestamp'] = pd.to_datetime(tags['timestamp'], unit='s')
        return tags
    
    def create_genre_matrix(self, movies_df):
        """Create a one-hot encoded genre matrix"""
        all_genres = set()
        for genres in movies_df['genres']:
            all_genres.update(genres)
        
        genre_matrix = pd.DataFrame(0, index=movies_df.index, 
                                  columns=list(all_genres))
        
        for idx, genres in enumerate(movies_df['genres']):
            genre_matrix.loc[idx, genres] = 1
            
        return genre_matrix
    
    def create_sparse_user_movie_matrix(self, ratings_df):
        """Create a sparse user-movie rating matrix"""
        # Create mappings for users and movies to matrix indices
        unique_users = ratings_df['userId'].unique()
        unique_movies = ratings_df['movieId'].unique()
        
        user_to_index = {user: i for i, user in enumerate(unique_users)}
        movie_to_index = {movie: i for i, movie in enumerate(unique_movies)}
        
        # Convert user and movie IDs to matrix indices
        user_indices = ratings_df['userId'].map(user_to_index)
        movie_indices = ratings_df['movieId'].map(movie_to_index)
        
        # Create sparse matrix
        sparse_matrix = csr_matrix(
            (ratings_df['rating'], (user_indices, movie_indices)),
            shape=(len(unique_users), len(unique_movies))
        )
        
        return sparse_matrix, user_to_index, movie_to_index

def main():
    # Use your exact filepath
    data_path = '/Users/danielscarnavack/MovieRec/ml-25m'
    
    loader = MovieLensLoader(data_path)
    
    try:
        print("\nLoading ratings...")
        ratings_df = loader.load_ratings()
        print("\nLoading movies...")
        movies_df = loader.load_movies()
        print("\nLoading tags...")
        tags_df = loader.load_tags()
        
        print("\nCreating matrices...")
        genre_matrix = loader.create_genre_matrix(movies_df)
        sparse_matrix, user_map, movie_map = loader.create_sparse_user_movie_matrix(ratings_df)
        
        print(f"\nSuccessfully loaded:")
        print(f"- {len(ratings_df)} ratings")
        print(f"- {len(movies_df)} movies")
        print(f"- {len(tags_df)} tags")
        
        # Print some statistics about the sparse matrix
        print(f"\nSparse matrix shape: {sparse_matrix.shape}")
        print(f"Sparse matrix density: {sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1]):.4%}")
        print(f"Memory usage: {sparse_matrix.data.nbytes / 1024 / 1024:.2f} MB")
        
        return ratings_df, movies_df, tags_df, genre_matrix, sparse_matrix, user_map, movie_map
    
    except FileNotFoundError as e:
        print(f"\nError: Could not find required data files!")
        print(f"Specific error: {e}")
        return None
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        return None

if __name__ == "__main__":
    main()
