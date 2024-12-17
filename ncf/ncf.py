import pandas as pd
import torch
from sklearn import preprocessing
import joblib
from ncf import RecSysModel

RATINGS_DIR = 'resources/data/train_val_test'
CHECKPOINT_PATH = 'resources/checkpoints'

batch_size = 512
layers = [32, 16, 8]
reg_layers = [0.3, 0.3, 0.3]
learning_rate = 5e-4
epochs = 40

k = 25
threshold = 6

df_test = pd.read_csv(f'{RATINGS_DIR}/ratings_test.csv', header=None, dtype='int32')
df_test.columns = ['uid', 'fid', 'rating']

label_enc_uid: preprocessing.LabelEncoder = joblib.load(f'{CHECKPOINT_PATH}/label_enc_uid.pkl')
label_enc_fid: preprocessing.LabelEncoder = joblib.load(f'{CHECKPOINT_PATH}/label_enc_fid.pkl')
model = RecSysModel(
    num_users=len(label_enc_uid.classes_),
    num_items=len(label_enc_fid.classes_),
    layers=layers,
    reg_layers=reg_layers
)

model.load_state_dict(
    torch.load(f'{CHECKPOINT_PATH}/ncf.pth', weights_only=True),
)

all_movies = df_test['fid'].unique()


def recommend_top_movies(user_id: int, model = model, label_enc_uid = label_enc_fid, label_enc_fid = label_enc_fid, all_movies = all_movies, k=25):
    model.eval()
    seen_movies = set(df_test[df_test['uid'] == user_id]['fid'])
    unseen_movies = [m for m in all_movies if m not in seen_movies]
    unseen_movies = label_enc_fid.transform(unseen_movies)
    predictions = []
    
    with torch.no_grad():
        batch_user_id = [user_id] * len(unseen_movies)
        batch_user_id = label_enc_uid.transform(batch_user_id)
        user_tensor = torch.tensor(batch_user_id)

        movie_tensor = torch.tensor(unseen_movies)
        predicted_ratings = model(user_tensor, movie_tensor).view(-1).tolist()
        # print(predicted_ratings)

        unseen_movies = label_enc_fid.inverse_transform(unseen_movies)
        # print(unseen_movies)
        predictions.extend(zip(unseen_movies, predicted_ratings))

    predictions.sort(key=lambda x: x[1], reverse=True)
    # print(predictions)
    top_k_movies = [movie_id for movie_id, _ in predictions[:k]]
    return top_k_movies