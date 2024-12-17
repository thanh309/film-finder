import os
import subprocess
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack,csr_matrix,vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

KNN_MODEL_PATH = 'content-based/knn_model.joblib'
if not os.path.exists(KNN_MODEL_PATH):
    print(f"{KNN_MODEL_PATH} does not exist. Running content-based/cosine_matrix.py to generate the model.")
    
    # Chạy script cosine_matrix.py để tạo model
    subprocess.run(['python', 'content-based/cosine_matrix.py'], check=True)
    
    # Sau khi script chạy xong, bạn có thể load lại model
    knn_loaded = joblib.load(KNN_MODEL_PATH)
    print("KNN model loaded successfully.")
else:
    # Nếu file đã tồn tại, chỉ cần load model
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

def cosine_distance(vec1, vec2):
    vec1 = np.array(vec1.toarray()).flatten() 
    vec2 = np.array(vec2.toarray()).flatten()
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return 1 - similarity

def all_watched_fid(user_id, ratings_data, k = 5):
    user_data = ratings_data[ratings_data['user_id'] == user_id]
    if len(user_data['fid'].tolist()) <5:
        k = len(user_data['fid'].tolist())
    return user_data['fid'].tolist(), k 

def predict_film_unwatch(user_id, ratings_data=ratings_data, knn=knn_loaded, 
                         combined_features_matrix=combined_features_matrix, 
                         all_film_id=all_film_id, film_id_to_index=film_id_to_index, 
                         prefer_point=6, n_recommendations=20):
    recommendations = []
    user_watched_fids,k = all_watched_fid(user_id, ratings_data)
    user_ratings = ratings_data[ratings_data['user_id'] == user_id]
    watched_high_rated = user_ratings[user_ratings['rating'] > prefer_point]
    
    
    if len(watched_high_rated) == 0:
        return recommendations
    prefer_indices = [film_id_to_index[fid] for fid in watched_high_rated['fid'].values if fid in film_id_to_index]
    
    ######## get feature vectors for all prefer film
    prefer_combined_features_matrices = []
    for i in prefer_indices:
        prefer_combined_features_matrices.append(combined_features_matrix[i])
       
    # print(prefer_combined_features_matrices)
    # mean_vector = prefer_combined_features_matrices.mean(axis=0)
    # print('Shape prefer:', prefer_combined_features_matrices[0].shape, type(prefer_combined_features_matrices[0]))
    combined_matrix = vstack(prefer_combined_features_matrices)
    # print('Shape combined:', combined_matrix.shape[0], type(combined_features_matrix[0]))

    # Bước 3: Tính trung bình trên các hàng (axis=0)
    mean_vector_dense = combined_matrix.mean(axis=0)  # Kết quả là numpy.matrix
    mean_vector = csr_matrix(mean_vector_dense) 

    # print("Shape mean:", mean_vector.shape, type(mean_vector))
    suggested_film_ids = set()
    n_neighbors = n_recommendations
    
    ######## get features_vectors for all fim watched
    all_watched_features_matrices = []
    all_watched_indices = [film_id_to_index[fid] for fid in user_watched_fids if fid in film_id_to_index]
    for i in all_watched_indices:
        all_watched_features_matrices.append(combined_features_matrix[i])
    
    while len(recommendations) < n_recommendations:
        distances, recommended_indices = knn.kneighbors(mean_vector, n_neighbors=n_neighbors, return_distance=True)
        
        for idx in recommended_indices[0]:
            film_id = all_film_id[idx]
            
            if film_id not in user_watched_fids and film_id not in suggested_film_ids:
                film_vector = combined_features_matrix[idx]
                
                cosine_distances = [
                    cosine_distance(film_vector, watched_vector) for watched_vector in all_watched_features_matrices
                ]

                nearest_k_indices = np.argsort(cosine_distances)[:k]
                nearest_k_ratings = [
                    user_ratings[user_ratings['fid'] == user_watched_fids[i]]['rating'].iloc[0] 
                    for i in nearest_k_indices
                ]
                mean_rating = np.mean(nearest_k_ratings)

                recommendations.append((film_id, mean_rating))
                suggested_film_ids.add(film_id)

        if len(recommendations) < n_recommendations:
            n_neighbors *= 2  

        if n_neighbors > len(all_film_id):
            break

    return recommendations
for i in range(1):
    user = ratings_dataframes[2]['user_id'].iloc[i]  # Lấy user_id từ ratings_test.csv
    recommendations = predict_film_unwatch(user)
    print(f"User {user} recommendations: {recommendations}")