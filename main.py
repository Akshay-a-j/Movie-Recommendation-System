import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

class MovieRecommender:
    def __init__(self, n_tables=5, n_projections=50):
        self.n_tables = n_tables
        self.n_projections = n_projections
        self.movies_df = None
        self.movie_features = None
        self.hash_tables = None
        self.projection_vectors = None

    def fit(self, movies_path: str):
        self.movies_df = pd.read_csv(movies_path)
        
        # Create feature matrix combining genres and titles
        genres_matrix = pd.get_dummies(
            self.movies_df['genres'].str.get_dummies('|')
        ).values
        
        vectorizer = TfidfVectorizer(max_features=300, stop_words='english')
        title_features = vectorizer.fit_transform(self.movies_df['title']).toarray()
        
        self.movie_features = np.hstack([genres_matrix, title_features])
        
        # Normalize features
        norms = np.linalg.norm(self.movie_features, axis=1)
        self.movie_features = self.movie_features / norms[:, np.newaxis]
        
        # Initialize LSH structures
        self.projection_vectors = [
            np.random.randn(self.n_projections, self.movie_features.shape[1]) 
            for _ in range(self.n_tables)
        ]
        
        self.hash_tables = [defaultdict(list) for _ in range(self.n_tables)]
        
        # Insert all movies into hash tables
        for movie_idx, movie_vector in enumerate(self.movie_features):
            for table_idx in range(self.n_tables):
                hash_value = self._hash_vector(movie_vector, table_idx)
                self.hash_tables[table_idx][hash_value].append(movie_idx)

    def _hash_vector(self, vector, table_idx):
        projections = np.dot(self.projection_vectors[table_idx], vector)
        threshold = np.percentile(projections, 50)
        return tuple(projections > threshold)

    def find_similar_movies(self, movie_id: int, k: int = 5):
        movie_idx = self.movies_df[self.movies_df['movieId'] == movie_id].index[0]
        vector = self.movie_features[movie_idx]
        
        candidates = set()
        for table_idx in range(self.n_tables):
            hash_value = self._hash_vector(vector, table_idx)
            candidates.update(self.hash_tables[table_idx][hash_value])
            
            # Get neighboring buckets
            hash_list = list(hash_value)
            for i in range(len(hash_list)):
                hash_list[i] = not hash_list[i]
                neighbor_hash = tuple(hash_list)
                candidates.update(self.hash_tables[table_idx][neighbor_hash])
                hash_list[i] = not hash_list[i]
        
        candidates.discard(movie_idx)
        
        if not candidates:
            query_genres = set(self.movies_df.iloc[movie_idx]['genres'].split('|'))
            candidates = set()
            for idx, row in self.movies_df.iterrows():
                if idx != movie_idx:
                    movie_genres = set(row['genres'].split('|'))
                    if len(query_genres.intersection(movie_genres)) > 0:
                        candidates.add(idx)
        
        distances = []
        for idx in candidates:
            distance = np.linalg.norm(vector - self.movie_features[idx])
            distances.append((distance, idx))
        
        return [idx for _, idx in sorted(distances)[:k]]
