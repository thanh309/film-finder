import os
import subprocess
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack,csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

KNN_MODEL_PATH = 'resources/checkpoints/knn_model.joblib'

knn_loaded = joblib.load(KNN_MODEL_PATH)
print("KNN model already exists and loaded successfully.")

FILM_PATH = 'resources/cleaned_data/cleaned_data.csv'
RATINGS_PATH ='resources/data/train_val_test'


rating_files = ['ratings_train.csv','ratings_val.csv','ratings_test.csv']
ratings_dataframes = []
for file in rating_files:
    file_path = os.path.join(RATINGS_PATH, file)
    try:
        df = pd.read_csv(file_path, delimiter=',', header=None, names=['user_id','fid', 'rating'])
        ratings_dataframes.append(df)
    except Exception as e:
        print(f"Error reading {file}: {e}")
ratings_data = pd.concat(ratings_dataframes[0:2], ignore_index=True)

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

all_film_id = film_data['fid'].tolist()
film_id_to_index = {fid: idx for idx, fid in enumerate(all_film_id)}


def all_watched_fid(user_id, ratings_data):
    user_data = ratings_data[ratings_data['user_id'] == user_id]
    return user_data['fid'].tolist()

def predict_film_unwatch(user_id, ratings_data=ratings_data, knn=knn_loaded,combined_features_matrix = combined_features_matrix,all_film_id = all_film_id,film_id_to_index=film_id_to_index, n_recommendations = 20):
    recommendations = []
    user_watched_fids = all_watched_fid(user_id, ratings_data)
    user_watched_indices = [film_id_to_index[fid] for fid in user_watched_fids if fid in film_id_to_index]
    
    for idx in user_watched_indices:
        film_vector = combined_features_matrix[idx]

        distances, indices = knn.kneighbors(film_vector, n_neighbors=n_recommendations + 1)
        
        recommended_fids = [all_film_id[i] for i in indices.flatten()]
        
        recommendations.extend([rec for rec in recommended_fids if rec != all_film_id[idx]])
    unique_recommendations = list(set(recommendations) - set(user_watched_fids))
    
    return unique_recommendations[:n_recommendations]

# for i in range(3):
#     user = ratings_dataframes[2]['user_id'].iloc[i]  # Lấy user_id từ ratings_test.csv
#     recommendations = predict_film_unwatch(user)
#     print(f"User {user} recommendations: {recommendations}")