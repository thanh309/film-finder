import pandas as pd 
import numpy as np 
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances
from utilities import * 

rounding_func = np.vectorize(rounding)

'''
In this approach, I use dataframe with only 3 columns: UserID, MovieID, Rating
For example: 
    UserID  |   MovieID  | Rating 
         1            1         5
         1            2         3
         1            3         4
         1            4         3
         1            5         3
'''

class NeighborhoodCF:
    def __init__(self, utility_matrix: np.ndarray, k_neighbors: int =10, uu_cf: bool =True, cosine: bool =True) -> None:
        self.utility_matrix = utility_matrix
        self.normalized_utility_matrix = normalize(self.utility_matrix)
        self.mean_ratings = find_nonzero_mean_ratings(self.utility_matrix)
        self.k_neighbors = k_neighbors
        self.uu_cf = uu_cf      # if uu_cf = True, we are building the user-user CF; otherwise item-item CF
        self.cosine = cosine    # if cosine = True, we use the cosine similarity; otherwise, we use pearson correlation
    
        
    def cosine_similarity(self, normalized_utility_matrix) -> np.ndarray:
        sim = 1 - pairwise_distances(normalized_utility_matrix, metric='cosine')
        sim[np.isnan(sim)] = 0  # Replace NaN with 0
        return sim
    
    
    def pearson_correlation(self, normalized_utility_matrix) -> np.ndarray:
        corr = np.corrcoef(normalized_utility_matrix)
        corr[np.isnan(corr)] = 0  # Replace NaN with 0
        return corr

    
    def predict_ratings(self) -> np.ndarray:
        pred = np.zeros_like(self.normalized_utility_matrix, dtype=float)
        if self.uu_cf:
            similarity_matrix = (
            self.cosine_similarity(self.normalized_utility_matrix) if self.cosine else self.pearson_correlation(self.normalized_utility_matrix)
            )
            for u in range(similarity_matrix.shape[0]):
                top_k_neighbors = np.argsort(similarity_matrix[u, :])[-self.k_neighbors:][::-1]
                
                sim_sum = np.sum(np.abs(similarity_matrix[u, top_k_neighbors]))
                sim_sum = max(sim_sum, 1e-8)  # Avoid division by 0
                
                for m in range(self.normalized_utility_matrix.shape[1]):
                    if self.normalized_utility_matrix[u, m] == 0:
                        weighted_sum = np.dot(
                            similarity_matrix[u, top_k_neighbors],
                            self.normalized_utility_matrix[top_k_neighbors, m]
                        )
                        pred[u, m] = self.mean_ratings[u] + (weighted_sum / sim_sum)
        else:
            similarity_matrix = (
            self.cosine_similarity(self.normalized_utility_matrix.T) if self.cosine else self.pearson_correlation(self.normalized_utility_matrix.T)
            )
            for m in range(similarity_matrix.shape[1]):
                top_k_neighbors = np.argsort(similarity_matrix[:, m])[-self.k_neighbors:][::-1]
                
                sim_sum = np.sum(np.abs(similarity_matrix[top_k_neighbors, m]))
                sim_sum = max(sim_sum, 1e-8)  # Avoid division by 0
                
                for u in range(self.normalized_utility_matrix.shape[0]):
                    if self.normalized_utility_matrix[u, m] == 0:
                        weighted_sum = np.dot(
                            similarity_matrix[top_k_neighbors, m],
                            self.normalized_utility_matrix[u, top_k_neighbors]
                        )
                        pred[u, m] = self.mean_ratings[u] + (weighted_sum / sim_sum)
        # Replace NaN with 0 in prediction matrix
        pred[np.isnan(pred)] = 0
        return pred, len(similarity_matrix)

        
        
    def recommend(self, id: int, predicted_ratings: np.ndarray, movies: pd.DataFrame, users: pd.DataFrame, top_n:int =5) -> list:
        if self.uu_cf:      # here, id is the user_id, we are recommend movies for user
            user_id = id 
            vectorized_index_to_movies_id = index_to_movies_id_vect(movies)
            vectorized_users_id_to_index = users_id_to_idex_vect(users)
            
            user_idx = int(vectorized_users_id_to_index(user_id))
            user_rated_indices = np.where(self.utility_matrix[user_idx, :] > 0)[0]
            user_pred = predicted_ratings[user_idx, :].copy()
            user_pred[user_rated_indices] = -1e8    # exclude watched items
            
            top_movies_idx = np.argsort(user_pred)[-top_n:][::-1]
            
            recommend_movies = movies[movies['MovieID'].isin(vectorized_index_to_movies_id(top_movies_idx))].copy()
            recommend_movies['predicted_score'] = rounding_func(user_pred[top_movies_idx])
            recommend_movies = recommend_movies.sort_values('predicted_score', ascending=False)
            
            return list(zip(recommend_movies['MovieID'].tolist(), recommend_movies['predicted_score'].tolist()))

        else: 
            movie_id = id 
            vectorized_index_to_users_id = index_to_users_id_vect(users)
            vectorized_movies_id_to_index = movies_id_to_idex_vect(movies)

            movie_idx = int(vectorized_movies_id_to_index(movie_id))
            movie_rated_indices = np.where(self.utility_matrix[:, movie_idx] > 0)[0]
            movie_pred = predicted_ratings[:, movie_idx].copy()
            movie_pred[movie_rated_indices] = -1e8      # exclude watched items
            
            top_users_idx = np.argsort(movie_pred)[-top_n:][::-1]

            recommend_users = users[users['UserID'].isin(vectorized_index_to_users_id(top_users_idx))].copy()
            recommend_users['predicted_score'] = rounding_func(movie_pred[top_users_idx])
            recommend_users = recommend_users.sort_values('predicted_score', ascending=False)
            
            return list(zip(recommend_users['UserID'].tolist(), recommend_users['predicted_score'].tolist()))

    
    def print_recommendation(self, predicted_ratings: np.ndarray, movies: pd.DataFrame, users: pd.DataFrame) -> None:
        print('Recommendation')
        print('------------------------------------------------------')
        if self.uu_cf: 
            vectorized_index_to_users_id = index_to_users_id_vect(users)
            for u in range(self.utility_matrix.shape[0]):
                user_id = vectorized_index_to_users_id(u)
                print(f'***** Recomended Movies for User {user_id}: *****')
                recommended = self.recommend(user_id, predicted_ratings=predicted_ratings, movies=movies, users=users, top_n=5)
                for movie_id, rating in recommended: 
                    print(f"Movie {movie_id}, rating: {rating}")
        
                print('------------------------------------------------------')
                
        else: 
            vectorized_index_to_movies_id = index_to_movies_id_vect(movies)
            for m in range(self.utility_matrix.shape[1]):
                movie_id = vectorized_index_to_movies_id(m)
                print(f'***** Recomended Users for Movie {movie_id}: *****')
                recommended = self.recommend(movie_id, predicted_ratings=predicted_ratings, movies=movies, users=users, top_n=5)
                for user_id, rating in recommended: 
                    print(f"User {user_id}, rating: {rating}")
                
                print('------------------------------------------------------')
                
        
        

def build_utility_matrix(df: pd.DataFrame) -> np.ndarray:
    utility_matrix = csr_matrix(df.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0).values)
    return utility_matrix.toarray()


def find_nonzero_mean_ratings(utility_matrix: np.ndarray) -> np.ndarray:
    mean_ratings = np.true_divide(
        utility_matrix.sum(axis=1),
        (utility_matrix != 0).sum(axis=1),
        where=(utility_matrix != 0).sum(axis=1) != 0
    )
    
    # Replace NaN with 0 for users with no ratings
    mean_ratings[np.isnan(mean_ratings)] = 0
    
    return mean_ratings

    

def normalize(utility_matrix: np.ndarray) -> np.ndarray:
    mean_ratings = find_nonzero_mean_ratings(utility_matrix)
    normalized_utility_matrix = utility_matrix.copy().astype(float)
    for u in range(normalized_utility_matrix.shape[0]):
        mask = normalized_utility_matrix[u, :] != 0
        normalized_utility_matrix[u, mask] -= mean_ratings[u]
    
    # Replace any remaining NaN with 0
    normalized_utility_matrix[np.isnan(normalized_utility_matrix)] = 0
    
    return normalized_utility_matrix
