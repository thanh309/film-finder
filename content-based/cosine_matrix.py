import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix,hstack
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# FILM_PATH = '../resources/cleaned_data/cleaned_data.csv'


FILM_PATH = 'resources/cleaned_data/cleaned_data.csv'
data = pd.read_csv(FILM_PATH)

data = pd.read_csv(FILM_PATH)
columns_to_keep = ['fid', 'contentRating', 
                   'genre', 'keywords', 'duration', 'actor', 'director']

film_data = data[columns_to_keep]

combined_features = film_data['genre']+ " " + film_data['director'] + " " + film_data['actor'] + " " + film_data['keywords']
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
one_hot_encoder = OneHotEncoder(sparse_output=True)
content_ratings_encoded = one_hot_encoder.fit_transform(film_data[['contentRating']])

scaler = MinMaxScaler()
duration_normalized = scaler.fit_transform(film_data[['duration']])  # Normalize to range [0, 1]
# Convert to sparse matrix
duration_sparse = csr_matrix(duration_normalized)

combined_features_matrix = hstack([feature_vectors, content_ratings_encoded,duration_normalized])
combined_features_matrix = csr_matrix(combined_features_matrix)

similarity = cosine_similarity(combined_features_matrix, combined_features_matrix)

np.save(os.path.join('content-based/film_similarity.npy'), similarity)
print("Similarity matrix saved successfully!")

all_film_id = film_data['fid'].tolist()
film_id_to_index = {fid: idx for idx, fid in enumerate(all_film_id)}

#train
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(combined_features_matrix)

KNN_MODEL_PATH = os.path.join('content-based', 'knn_model.joblib')
joblib.dump(knn, KNN_MODEL_PATH)

print(f"KNN model saved successfully at {KNN_MODEL_PATH}")