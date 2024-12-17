import pandas as pd
import torch
from sklearn import preprocessing
import joblib
import torch.nn as nn

class RecSysModel(nn.Module):
    def __init__(self, num_users, num_items, layers, reg_layers):
        super(RecSysModel, self).__init__()
        self.num_layers = len(layers)

        self.user_embedding = nn.Embedding(num_users, layers[0] // 2)
        self.item_embedding = nn.Embedding(num_items, layers[0] // 2)

        fc_layers = []
        input_size = layers[0]
        for i in range(1, len(layers)):
            fc_layers.append(nn.Linear(input_size, layers[i]))
            fc_layers.append(nn.ReLU())
            if reg_layers[i] > 0:
                fc_layers.append(nn.Dropout(reg_layers[i]))
            input_size = layers[i]

        self.fc_layers = nn.Sequential(*fc_layers)
        self.output_layer = nn.Linear(layers[-1], 1)

        self._init_weight()

    def forward(self, user_input, item_input):
        # [batch, num_users, embed_size]
        user_latent = self.user_embedding(user_input)
        item_latent = self.item_embedding(item_input)

        vector = torch.cat([user_latent, item_latent], dim=-1)

        vector = self.fc_layers(vector)

        prediction = self.output_layer(vector)
        return prediction

    def _init_weight(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

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


def recommend_top_movies(user_id: int, model = model, label_enc_uid = label_enc_uid, label_enc_fid = label_enc_fid, all_movies = all_movies, k=25):
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


