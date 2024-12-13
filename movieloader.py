import pandas as pd
import numpy as np
from pathlib import Path
import gdown
import os

class MovieLensLoader:
    def __init__(self):
        # Replace these with your Google Drive file IDs
        self.ratings_file_id = "1eRSTmyYGucnToFuah2G_JV10fd4wZhxv"
        self.movies_file_id = "1M7Hx1nSH7F4Ik-lqq0hb_nl-deEzl4H3"
        
        # Create cache directory if it doesn't exist
        self.cache_dir = Path('cache')
        self.cache_dir.mkdir(exist_ok=True)
        
    def download_file(self, file_id, output_path):
        """Download file from Google Drive"""
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, str(output_path), quiet=False)
    
    def load_ratings(self):
        """Load and preprocess ratings data"""
        ratings_path = self.cache_dir / 'ratings.csv'
        
        # Download if not in cache
        if not ratings_path.exists():
            self.download_file(self.ratings_file_id, ratings_path)
        
        ratings = pd.read_csv(ratings_path)
        ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
        return ratings
    
    def load_movies(self):
        """Load and preprocess movies data"""
        movies_path = self.cache_dir / 'movies.csv'
        
        # Download if not in cache
        if not movies_path.exists():
            self.download_file(self.movies_file_id, movies_path)
        
        movies = pd.read_csv(movies_path)
        movies['genres'] = movies['genres'].str.split('|')
        return movies
