import pandas as pd 
import numpy as np 
import math 

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

def build_vector(df: pd.DataFrame) -> tuple: 
    index_to_df_id = {}
    df_id_to_index = {}
    
    if 'MovieID' in df.columns:
        for i in range(len(df)): 
            index_to_df_id[i] = int(df.MovieID.iloc[i])
            df_id_to_index[int(df.MovieID.iloc[i])] = i
    else: 
        for i in range(len(df)): 
            index_to_df_id[i] = int(df.UserID.iloc[i])
            df_id_to_index[int(df.UserID.iloc[i])] = i
    
    vectorized_index_to_df_id = np.vectorize(index_to_df_id.get)
    vectorized_df_id_to_index = np.vectorize(df_id_to_index.get)
    
    return (vectorized_index_to_df_id, vectorized_df_id_to_index)
    
    
def index_to_movies_id_vect(movies: pd.DataFrame) -> np.vectorize:
    return build_vector(movies)[0]


def movies_id_to_idex_vect(movies: pd.DataFrame) -> np.vectorize:
    return build_vector(movies)[1]


def index_to_users_id_vect(users: pd.DataFrame) -> np.vectorize: 
    return build_vector(users)[0]


def users_id_to_idex_vect(users: pd.DataFrame) -> np.vectorize: 
    return build_vector(users)[1]


def rounding(rating):
    if rating - int(rating) < 0.5: 
        return int(rating)
    
    return math.ceil(rating)