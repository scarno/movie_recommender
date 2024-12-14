import pandas as pd
import numpy as np
from pathlib import Path
import gdown
import os
import time

class MovieLensLoader:
    def __init__(self):
        self.ratings_file_id = "1eRSTmyYGucnToFuah2G_JV10fd4wZhxv"
        self.movies_file_id = "1M7Hx1nSH7F4Ik-lqq0hb_nl-deEzl4H3"
        
        # Create cache directory
        self.cache_dir = Path('cache')
        self.cache_dir.mkdir(exist_ok=True)
    
    def download_file(self, file_id, output_path, max_retries=3):
        """Download file from Google Drive with retries"""
        url = f'https://drive.google.com/uc?id={file_id}'
        
        for attempt in range(max_retries):
            try:
                if not output_path.exists():
                    gdown.download(url, str(output_path), quiet=False)
                    time.sleep(1)  # Add small delay between downloads
                
                # Verify file exists and has size
                if output_path.exists() and output_path.stat().st_size > 0:
                    return True
                    
            except Exception as e:
                print(f"Download attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to download file after {max_retries} attempts")
                time.sleep(2)  # Wait before retrying
        
        return False
    
    def load_ratings(self):
        """Load and preprocess ratings data"""
        ratings_path = self.cache_dir / 'ratings.csv'
        
        # Ensure ratings file is downloaded
        self.download_file(self.ratings_file_id, ratings_path)
        
        try:
            # Use chunks for large file
            chunks = []
            for chunk in pd.read_csv(ratings_path, chunksize=100000):
                chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], unit='s')
                chunks.append(chunk)
            ratings = pd.concat(chunks)
            return ratings
            
        except Exception as e:
            raise Exception(f"Error loading ratings data: {str(e)}")
    
    def load_movies(self):
        """Load and preprocess movies data"""
        movies_path = self.cache_dir / 'movies.csv'
        
        # Ensure movies file is downloaded
        self.download_file(self.movies_file_id, movies_path)
        
        try:
            movies = pd.read_csv(movies_path)
            movies['genres'] = movies['genres'].str.split('|')
            return movies
            
        except Exception as e:
            raise Exception(f"Error loading movies data: {str(e)}")
        
    def clear_cache(self):
        """Clear cached files"""
        try:
            for file in self.cache_dir.glob('*'):
                file.unlink()
        except Exception as e:
            print(f"Error clearing cache: {str(e)}")
