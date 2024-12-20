import pandas as pd 
import numpy as np 
import joblib 
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import root_mean_squared_error
from utilities import * 

from collections import defaultdict
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
    
    
    def predict_user_ratings(self, user_id: int, users: pd.DataFrame) -> np.ndarray:
        """
        Predict ratings for all movies for a single user (for user-user CF).
        """
        vectorized_users_id_to_index = users_id_to_index_vect(users)
        user_idx = int(vectorized_users_id_to_index(user_id))

        # Compute user-user similarity
        if self.cosine:
            similarity_matrix = self.cosine_similarity(self.normalized_utility_matrix)
        else:
            similarity_matrix = self.pearson_correlation(self.normalized_utility_matrix)

        user_sims = similarity_matrix[user_idx, :]
        top_k_neighbors = np.argsort(user_sims)[-self.k_neighbors:][::-1]

        user_pred = np.zeros(self.utility_matrix.shape[1], dtype=float)

        sim_sum = np.sum(np.abs(user_sims[top_k_neighbors]))
        sim_sum = max(sim_sum, 1e-8)  # Avoid division by zero

        # Compute predicted ratings for all movies for this user
        # pred(u,m) = mean_ratings[u] + ( sum(sim(u,neighbors)*norm_utility[neighbors,m]) / sum(|sim|) )
        weighted_sums = user_sims[top_k_neighbors].dot(self.normalized_utility_matrix[top_k_neighbors, :])
        user_pred = self.mean_ratings[user_idx] + (weighted_sums / sim_sum)

        # Replace NaN if any
        user_pred[np.isnan(user_pred)] = 0
        return user_pred
    

    def predict_movie_ratings(self, movie_id: int, movies: pd.DataFrame) -> np.ndarray:
        """
        Predict ratings for all users for a single movie (for item-item CF).
        """
        vectorized_movies_id_to_index = movies_id_to_index_vect(movies)
        movie_idx = int(vectorized_movies_id_to_index(movie_id))

        # Compute item-item similarity
        if self.cosine:
            similarity_matrix = self.cosine_similarity(self.normalized_utility_matrix.T)
        else:
            similarity_matrix = self.pearson_correlation(self.normalized_utility_matrix.T)

        movie_sims = similarity_matrix[movie_idx, :]
        top_k_neighbors = np.argsort(movie_sims)[-self.k_neighbors:][::-1]

        sim_sum = np.sum(np.abs(movie_sims[top_k_neighbors]))
        sim_sum = max(sim_sum, 1e-8)

        # Compute predicted ratings for all users for this movie
        # pred(u,m) = mean_ratings[u] + ( sum(sim(m,neighbors)*norm_utility[u,neighbors]) / sum(|sim|) )
        weighted_sums = self.normalized_utility_matrix[:, top_k_neighbors].dot(movie_sims[top_k_neighbors])
        movie_pred = self.mean_ratings + (weighted_sums / sim_sum)

        movie_pred[np.isnan(movie_pred)] = 0
        return movie_pred
    
    
    def recommend(self, id: int, movies: pd.DataFrame, users: pd.DataFrame, top_n:int =25) -> list:
        if self.uu_cf:
            # Recommend movies for a given user
            user_id = id
            user_pred = self.predict_user_ratings(user_id, users)
            
            vectorized_index_to_movies_id = index_to_movies_id_vect(movies)
            vectorized_users_id_to_index = users_id_to_index_vect(users)
            
            user_idx = int(vectorized_users_id_to_index(user_id))

            # Exclude movies already rated
            user_rated_indices = np.where(self.utility_matrix[user_idx, :] > 0)[0]
            user_pred[user_rated_indices] = -1e8

            top_movies_idx = np.argsort(user_pred)[-top_n:][::-1]
            
            recommend_movies = movies[movies['MovieID'].isin(vectorized_index_to_movies_id(top_movies_idx))].copy()
            recommend_movies['predicted_score'] = rounding_func(user_pred[top_movies_idx])
            recommend_movies = recommend_movies.sort_values('predicted_score', ascending=False)

            return list(zip(recommend_movies['MovieID'].tolist(), recommend_movies['predicted_score'].tolist()))
        else:
            # Recommend users for a given movie
            movie_id = id
            movie_pred = self.predict_movie_ratings(movie_id, movies)

            vectorized_index_to_users_id = index_to_users_id_vect(users)
            vectorized_movies_id_to_index = movies_id_to_index_vect(movies)
            
            movie_idx = int(vectorized_movies_id_to_index(movie_id))

            # Exclude users who have already rated this movie
            movie_rated_indices = np.where(self.utility_matrix[:, movie_idx] > 0)[0]
            movie_pred[movie_rated_indices] = -1e8

            top_users_idx = np.argsort(movie_pred)[-top_n:][::-1]

            recommend_users = users[users['UserID'].isin(vectorized_index_to_users_id(top_users_idx))].copy()
            recommend_users['predicted_score'] = rounding_func(movie_pred[top_users_idx])
            recommend_users = recommend_users.sort_values('predicted_score', ascending=False)

            return list(zip(recommend_users['UserID'].tolist(), recommend_users['predicted_score'].tolist()))

        
    def recommend_using_predicted_ratings(self, id: int, predicted_ratings: np.ndarray, movies: pd.DataFrame, users: pd.DataFrame, top_n:int =25) -> list:
        if self.uu_cf:      # here, id is the user_id, we are recommend movies for user
            user_id = id 
            vectorized_index_to_movies_id = index_to_movies_id_vect(movies)
            vectorized_users_id_to_index = users_id_to_index_vect(users)
            
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
            vectorized_movies_id_to_index = movies_id_to_index_vect(movies)

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
                recommended = self.recommend_using_predicted_ratings(user_id, predicted_ratings=predicted_ratings, movies=movies, users=users, top_n=25)
                for movie_id, rating in recommended: 
                    print(f"Movie {movie_id}, rating: {rating}")
        
                print('------------------------------------------------------')
                
        else: 
            vectorized_index_to_movies_id = index_to_movies_id_vect(movies)
            for m in range(self.utility_matrix.shape[1]):
                movie_id = vectorized_index_to_movies_id(m)
                print(f'***** Recomended Users for Movie {movie_id}: *****')
                recommended = self.recommend_using_predicted_ratings(movie_id, predicted_ratings=predicted_ratings, movies=movies, users=users, top_n=25)
                for user_id, rating in recommended: 
                    print(f"User {user_id}, rating: {rating}")
                
                print('------------------------------------------------------')
                
 
class MatrixFactorizationCF: 
    def __init__(self, R: np.ndarray, K: int =20, learning_rate: float =0.005, epochs: int = 30, regularization: float = 0.02, uu_mf: bool =True, min_rating: float = 1.0, max_rating: float = 10.0) -> float:      
        self.R = R
        self.num_users, self.num_movies = self.R.shape
        self.K = K    # number of latent features 
        self.lr = learning_rate
        self.epochs = epochs 
        self.regularization = regularization
        self.uu_mf = uu_mf
        if self.uu_mf: 
            self.checkpoint = 'best_uumf_model.joblib' 
        else:
            self.checkpoint = 'best_iimf_model.joblib'
        self.min_rating = min_rating 
        self.max_rating = max_rating 
        
        # Initialize laten factors and biases
        self.P = None 
        self.Q = None 
        self.b_u = None 
        self.b_m = None 
        self.mu = None 
    
    def train(self, validation_data: np.ndarray =None) -> None:     
        # Initialize latent feature matrices 
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_movies, self.K))
        
        self.b_u = np.zeros(self.num_users)
        self.b_m = np.zeros(self.num_movies)
        self.mu = np.mean(self.R[self.R > 0])
        
        # Create training samples 
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_movies)
            if self.R[i, j] > 0
        ]
        
        best_val_rmse = float('inf')  # Track the best validation RMSE
        
        # Perform SGD for each epoch 
        for epoch in range(self.epochs): 
            np.random.shuffle(self.samples)
            total_loss = 0
            for i, j, r in self.samples: 
                # Predict current rating
                pred = self.predict_single(i, j, clip=False)
                e = r - pred 
                
                # Update biases 
                self.b_u[i] += self.lr * (e - self.regularization * self.b_u[i])
                self.b_m[j] += self.lr * (e - self.regularization * self.b_m[j])
                
                # Update latent features 
                P_i_old = self.P[i, :].copy()
                self.P[i, :] += self.lr * (e * self.Q[j, :] - self.regularization * self.P[i, :])
                self.Q[j, :] += self.lr * (e * P_i_old - self.regularization * self.Q[j, :])
                
                # Accumulate loss 
                total_loss += e**2 + self.regularization * (
                    np.sum(self.P[i, :] ** 2) + 
                    np.sum(self.Q[j, :] ** 2) + 
                    self.b_u[i] ** 2 + 
                    self.b_m[j] ** 2
                )
        
            train_rmse = np.sqrt(total_loss / len(self.samples))
            
            # Validation loss calculation 
            if validation_data is not None:
                val_loss = 0
                val_samples = [
                    (i, j, validation_data[i, j])
                    for i in range(self.num_users)
                    for j in range(self.num_movies)
                    if validation_data[i, j] > 0
                ]
                for i, j, r in val_samples:
                    pred = self.predict_single(i, j, clip=True)
                    e = r - pred
                    val_loss += e**2
                val_rmse = np.sqrt(val_loss / len(val_samples))
                
                print(f'Epoch: {epoch + 1} - Train RMSE: {train_rmse:.4f}, Validation RMSE: {val_rmse:.4f}')
                # Update best validation RMSE
                if val_rmse < best_val_rmse: 
                    best_val_rmse = val_rmse
                    
                    # Save the model with best RMSE
                    self.save_model(self.checkpoint)
            else:
                print(f'Epoch: {epoch + 1} - Train RMSE: {train_rmse:.4f}')
        
        return float(best_val_rmse)
    
    def predict_single(self, i: int, j: int, clip: bool =True) -> float: 
        pred = self.mu + self.b_u[i] + self.b_m[j] + np.dot(self.P[i, :], self.Q[j, :].T)
        if clip:
            pred = np.clip(pred, self.min_rating, self.max_rating)
        return pred
    
    
    def predict_user_ratings(self, user_id: int, users: pd.DataFrame) -> np.ndarray:
        """
        Predict ratings for all movies for a given user_id.
        """
        vectorized_users_id_to_index = users_id_to_index_vect(users)
        user_idx = int(vectorized_users_id_to_index(user_id))
        
        # Vectorized prediction for all movies: 
        # pred(u, m) = mu + b_u[u] + b_m[m] + P[u,:].dot(Q[m,:].T)
        user_pred = (self.mu 
                    + self.b_u[user_idx] 
                    + self.b_m 
                    + self.P[user_idx, :].dot(self.Q.T))
        
        # Clip predictions to the allowed range
        user_pred = np.clip(user_pred, self.min_rating, self.max_rating)
        return user_pred


    def predict_movie_ratings(self, movie_id: int, movies: pd.DataFrame) -> np.ndarray:
        """
        Predict ratings for all users for a given movie_id.
        """
        vectorized_movies_id_to_index = movies_id_to_index_vect(movies)
        movie_idx = int(vectorized_movies_id_to_index(movie_id))
        
        # Vectorized prediction for all users:
        # pred(u, m) = mu + b_u[u] + b_m[m] + P[u,:].dot(Q[m,:].T)
        # Here we fix m and vary u:
        movie_pred = (self.mu
                    + self.b_u
                    + self.b_m[movie_idx]
                    + self.P.dot(self.Q[movie_idx, :]))
        
        # Clip predictions to the allowed range
        movie_pred = np.clip(movie_pred, self.min_rating, self.max_rating)
        return movie_pred

    
    def full_prediction(self) -> np.ndarray:
        pred_matrix = self.mu + self.b_u[:, np.newaxis] + self.b_m[np.newaxis:, ] + self.P.dot(self.Q.T)
        pred_matrix = np.clip(pred_matrix, self.min_rating, self.max_rating)
        return pred_matrix
    
    
    def recommend(self, id: int, movies: pd.DataFrame, users: pd.DataFrame, top_n: int = 25) -> list:
        if self.uu_mf:
            # Here, 'id' is the user_id. We recommend movies for this user.
            user_id = id
            vectorized_index_to_movies_id = index_to_movies_id_vect(movies)
            user_pred = self.predict_user_ratings(user_id, users)
            
            vectorized_users_id_to_index = users_id_to_index_vect(users)
            user_idx = int(vectorized_users_id_to_index(user_id))
            
            # Exclude movies already rated by this user
            user_rated_indices = np.where(self.R[user_idx, :] > 0)[0]
            user_pred[user_rated_indices] = -1e8  # exclude watched items
            
            # Get top-N movies
            top_movies_idx = np.argsort(user_pred)[-top_n:][::-1]
            
            recommend_movies = movies[movies['MovieID'].isin(vectorized_index_to_movies_id(top_movies_idx))].copy()
            recommend_movies['predicted_score'] = rounding_func(user_pred[top_movies_idx])
            recommend_movies = recommend_movies.sort_values('predicted_score', ascending=False)
            
            return list(zip(recommend_movies['MovieID'].tolist(), recommend_movies['predicted_score'].tolist()))
        else:
            # Here, 'id' is the movie_id. We recommend users for this movie.
            movie_id = id
            vectorized_index_to_users_id = index_to_users_id_vect(users)
            movie_pred = self.predict_movie_ratings(movie_id, movies)
            
            vectorized_movies_id_to_index = movies_id_to_index_vect(movies)
            movie_idx = int(vectorized_movies_id_to_index(movie_id))
            
            # Exclude users who have already rated this movie
            movie_rated_indices = np.where(self.R[:, movie_idx] > 0)[0]
            movie_pred[movie_rated_indices] = -1e8  # exclude users who have seen this movie
            
            # Get top-N users
            top_users_idx = np.argsort(movie_pred)[-top_n:][::-1]

            recommend_users = users[users['UserID'].isin(vectorized_index_to_users_id(top_users_idx))].copy()
            recommend_users['predicted_score'] = rounding_func(movie_pred[top_users_idx])
            recommend_users = recommend_users.sort_values('predicted_score', ascending=False)
            
            return list(zip(recommend_users['UserID'].tolist(), recommend_users['predicted_score'].tolist()))
        
        
    def recommend_using_predicted_ratings(self, id: int, predicted_ratings: np.ndarray, movies: pd.DataFrame, users: pd.DataFrame, top_n:int =25) -> list:
        if self.uu_mf:      # here, id is the user_id, we are recommend movies for user
            user_id = id 
            vectorized_index_to_movies_id = index_to_movies_id_vect(movies)
            vectorized_users_id_to_index = users_id_to_index_vect(users)
            
            user_idx = int(vectorized_users_id_to_index(user_id))
            user_rated_indices = np.where(self.R[user_idx, :] > 0)[0]
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
            vectorized_movies_id_to_index = movies_id_to_index_vect(movies)

            movie_idx = int(vectorized_movies_id_to_index(movie_id))
            movie_rated_indices = np.where(self.R[:, movie_idx] > 0)[0]
            movie_pred = predicted_ratings[:, movie_idx].copy()
            movie_pred[movie_rated_indices] = -1e8      # exclude watched items
            
            top_users_idx = np.argsort(movie_pred)[-top_n:][::-1]

            recommend_users = users[users['UserID'].isin(vectorized_index_to_users_id(top_users_idx))].copy()
            recommend_users['predicted_score'] = rounding_func(movie_pred[top_users_idx])
            recommend_users = recommend_users.sort_values('predicted_score', ascending=False)
            
            return list(zip(recommend_users['UserID'].tolist(), recommend_users['predicted_score'].tolist()))

    
    def print_recommendation(self, predicted_R: np.ndarray, movies: pd.DataFrame, users: pd.DataFrame) -> None:
        print('Recommendation')
        print('------------------------------------------------------')
        if self.uu_mf: 
            vectorized_index_to_users_id = index_to_users_id_vect(users)
            for u in range(self.R.shape[0]):
                user_id = vectorized_index_to_users_id(u)
                print(f'***** Recomended Movies for User {user_id}: *****')
                recommended = self.recommend_using_predicted_ratings(user_id, predicted_R=predicted_R, movies=movies, users=users, top_n=25)
                for movie_id, rating in recommended: 
                    print(f"Movie {movie_id}, rating: {rating}")
        
                print('------------------------------------------------------')
                
        else: 
            vectorized_index_to_movies_id = index_to_movies_id_vect(movies)
            for m in range(self.R.shape[1]):
                movie_id = vectorized_index_to_movies_id(m)
                print(f'***** Recomended Users for Movie {movie_id}: *****')
                recommended = self.recommend_using_predicted_ratings(movie_id, predicted_R=predicted_R, movies=movies, users=users, top_n=25)
                for user_id, rating in recommended: 
                    print(f"User {user_id}, rating: {rating}")
                
                print('------------------------------------------------------')
                
                
    def save_model(self, filename: str) -> None:
        """
        Save the model parameters using joblib.
        """
        model_data = {
            "K": self.K,
            "learning_rate": self.lr,
            "epochs": self.epochs,
            "regularization": self.regularization,
            "uu_mf": self.uu_mf,
            "min_rating": self.min_rating,
            "max_rating": self.max_rating,
            "P": self.P,
            "Q": self.Q,
            "b_u": self.b_u,
            "b_m": self.b_m,
            "mu": self.mu
        }
        joblib.dump(model_data, filename)
        print(f"Model saved to {filename}.")
        
    
    @classmethod
    def load_model(cls, filename: str, R: np.ndarray):
        """
        Load the model parameters using joblib and return a new instance.
        """
        model_data = joblib.load(filename)
        
        # Reconstruct the model instance
        model = cls(
            R=R,
            K=model_data["K"],
            learning_rate=model_data["learning_rate"],
            epochs=model_data["epochs"],
            regularization=model_data["regularization"],
            uu_mf=model_data["uu_mf"],
            min_rating=model_data["min_rating"],
            max_rating=model_data["max_rating"]
        )
        
        # Load the trained parameters
        model.P = model_data["P"]
        model.Q = model_data["Q"]
        model.b_u = model_data["b_u"]
        model.b_m = model_data["b_m"]
        model.mu = model_data["mu"]

        print(f"Model loaded from {filename}.")
        return model
                

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


def train_validation_split(R: np.ndarray, validation_ratio: float = 0.2, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Split the rating matrix R into training and validation sets.
    
    Parameters:
    - R (np.ndarray): User-item rating matrix.
    - validation_ratio (float): Proportion of ratings to include in the validation set.
    - seed (int): Random seed for reproducibility.
    
    Returns:
    - train_R (np.ndarray): Training rating matrix.
    - validation_R (np.ndarray): Validation rating matrix.
    """
    np.random.seed(seed)
    train_R = R.copy()
    validation_R = np.zeros(R.shape)
    
    for user in range(R.shape[0]):
        rated_items = np.where(R[user, :] > 0)[0]
        if len(rated_items) == 0:
            continue  # Skip users with no ratings
        n_validation = max(1, int(len(rated_items) * validation_ratio))
        validation_items = np.random.choice(rated_items, size=n_validation, replace=False)
        train_R[user, validation_items] = 0
        validation_R[user, validation_items] = R[user, validation_items]
    
    return train_R, validation_R


def mf_hyperparameter_tuning(
    train_R: np.ndarray,
    hyperparameter_combinations: list,
    val_R: np.ndarray
) -> pd.DataFrame:
    
    results = []
    
    for idx, (K, lr, reg, epochs) in enumerate(hyperparameter_combinations):
        print(f"Evaluating combination {idx + 1}/{len(hyperparameter_combinations)}: K={K}, lr={lr}, reg={reg}, epochs={epochs}")
        
        # Initialize and train the model
        mf = MatrixFactorizationCF(
            R=train_R,
            K=K,
            learning_rate=lr,
            epochs=epochs,
            regularization=reg,
            uu_mf=True,  # or False, depending on your focus
            min_rating=1.0,
            max_rating=10.0
        )
        best_val_rmse = mf.train(val_R)
        
        # Record the results
        results.append({
            'K': K,
            'learning_rate': lr,
            'regularization': reg,
            'epochs': epochs,
            'validation_RMSE': best_val_rmse
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df


def neighborhood_hyperparameter_tuning(
    R: np.ndarray,
    hyperparameter_combinations: list,
    validation_ratio: float = 0.2
) -> pd.DataFrame:
    
    results = []
    
    # Split the data once to ensure consistency across hyperparameter evaluations
    train_R, validation_R = train_validation_split(R, validation_ratio=validation_ratio)
    
    for idx, (k, uu_cf, cosine) in enumerate(hyperparameter_combinations): 
        print(f"Evaluating combination {idx + 1}/{len(hyperparameter_combinations)}: k_neighbors={k}, uu_cf={uu_cf}, cosine={cosine}")
        
        # Initialize the model with current hyperparameters 
        neighborhood_cf = NeighborhoodCF(utility_matrix=train_R, k_neighbors=k, uu_cf=uu_cf, cosine=cosine)
        
        # Generate predictions
        predicted_R = neighborhood_cf.predict_ratings()[0]
        
        # Evaluate on validation set
        # Only consider non-zero entries in validation_R
        val_users, val_items = np.where(validation_R > 0)
        val_true = validation_R[val_users, val_items]
        val_pred = predicted_R[val_users, val_items]
        
        # Filter out NaN values
        valid_indices = ~np.isnan(val_true) & ~np.isnan(val_pred)
        val_true = val_true[valid_indices]
        val_pred = val_pred[valid_indices]
        
        # Calculate RMSE
        if len(val_true) > 0:  # Ensure no empty array
            rmse = root_mean_squared_error(val_true, val_pred) 
        else:
            rmse = float('inf')  # Handle edge case of no valid comparisons

        # Log results
        results.append({
            'k_neighbors': k,
            'uu_cf': uu_cf,
            'cosine': cosine,
            'rmse': rmse
        })
        
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df


def calculate_precision_recall(user_ratings, k, threshold):
    user_ratings.sort(key=lambda x: x[0], reverse=True)

    n_rel = sum(true_r >= threshold for _, true_r in user_ratings)
    n_rec_k = sum(est >= threshold for est, _ in user_ratings[:k])
    n_rel_and_rec_k = sum(
        (true_r >= threshold) and (est >= threshold) for est, true_r in user_ratings[:k]
    )

    precision = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
    recall = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precision, recall

def eval(
    predicted_ratings: pd.DataFrame, 
    test_data: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame
) -> dict: 
    
    # Map user and movie IDs to indices 
    vectorized_users_id_to_index = users_id_to_index_vect(users)
    vectorized_movies_id_to_index = movies_id_to_index_vect(movies)
    
    # Extract true rating and predicted rating 
    true_ratings = []
    pred_ratings = []
    
    mydict = defaultdict(list)
    
    for _, row in test_data.iterrows():
        user_id = row['UserID']
        movie_id = row['MovieID']
        true_rating = row['Rating']
        
        user_idx = int(vectorized_users_id_to_index(user_id))
        movie_idx = int(vectorized_movies_id_to_index(movie_id))
        
        pred_rating = predicted_ratings[user_idx, movie_idx]
        
        true_ratings.append(true_rating)
        pred_ratings.append(pred_rating)
        mydict[user_id].append((pred_rating, true_rating))
        
    user_ratings = list(zip(pred_ratings, true_ratings))
        
    rmse = root_mean_squared_error(true_ratings, pred_ratings)
    
    avg_precision = 0.0
    avg_recall = 0.0

    for user, user_ratings in mydict.items():
        precision, recall = calculate_precision_recall(user_ratings, k=25, threshold=6)
        avg_precision += precision
        avg_recall += recall
        
    avg_precision /= len(mydict)
    avg_recall /= len(mydict)
    f1_score = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall)
    
    
    return {'RMSE': float(rmse), 'Precision': float(avg_precision), 'Recall': float(avg_recall), 'F1-Score': float(f1_score)}