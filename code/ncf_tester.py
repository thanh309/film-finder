import pandas as pd
import torch
from sklearn import preprocessing
import joblib
from ncf import RecSysModel, MyDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn
from collections import defaultdict
from sklearn.metrics import root_mean_squared_error

RATINGS_DIR = 'resources/data/train_val_test'
CHECKPOINT_PATH = 'resources/checkpoints'

batch_size = 512
layers = [32, 16, 8]
reg_layers = [0.3, 0.3, 0.3]
learning_rate = 5e-4
epochs = 40

k = 25
threshold = 6

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

def recommend_top_movies(model, label_enc_uid, label_enc_fid, user_id: int, all_movies, seen_movies: set, device=device, k=25):
    model.eval()
    unseen_movies = [m for m in all_movies if m not in seen_movies]
    unseen_movies = label_enc_fid.transform(unseen_movies)
    predictions = []
    
    with torch.no_grad():
        batch_user_id = [user_id] * len(unseen_movies)
        batch_user_id = label_enc_uid.transform(batch_user_id)
        user_tensor = torch.tensor(batch_user_id).to(device) # [batch]

        movie_tensor = torch.tensor(unseen_movies).to(device) # [batch]
        predicted_ratings = model(user_tensor, movie_tensor).view(-1).tolist()
        # print(predicted_ratings)

        unseen_movies = label_enc_fid.inverse_transform(unseen_movies)
        # print(unseen_movies)
        predictions.extend(zip(unseen_movies, predicted_ratings))

    predictions.sort(key=lambda x: x[1], reverse=True)
    # print(predictions)
    top_k_movies = [movie_id for movie_id, _ in predictions[:k]]
    return top_k_movies


def test1():
    df_test = pd.read_csv(f'{RATINGS_DIR}/ratings_test.csv', header=None, dtype='int32')
    df_test.columns = ['uid', 'fid', 'rating']

    label_enc_uid: preprocessing.LabelEncoder = joblib.load(f'{CHECKPOINT_PATH}/label_enc_uid.pkl')
    label_enc_fid: preprocessing.LabelEncoder = joblib.load(f'{CHECKPOINT_PATH}/label_enc_fid.pkl')
    model = RecSysModel(
        num_users=len(label_enc_uid.classes_),
        num_items=len(label_enc_fid.classes_),
        layers=layers,
        reg_layers=reg_layers
    ).to(device)

    model.load_state_dict(
        torch.load(f'{CHECKPOINT_PATH}/ncf.pth', weights_only=True),
    )

    all_movies = df_test['fid'].unique()
    all_users = df_test['uid'].unique() # not needed, for testing purpose only

    print('start')
    for user_id in tqdm(all_users):
        seen_movies = set(df_test[df_test['uid'] == user_id]['fid'])

        recommendations = recommend_top_movies(
            model=model,
            label_enc_uid=label_enc_uid,
            label_enc_fid = label_enc_fid,
            user_id=user_id,
            all_movies=all_movies,
            seen_movies=seen_movies,
        )

        # print(recommendations)

def eval():
    df_test = pd.read_csv(f'{RATINGS_DIR}/ratings_test.csv', header=None, dtype='int32')
    df_test.columns = ['uid', 'fid', 'rating']

    label_enc_uid: preprocessing.LabelEncoder = joblib.load(f'{CHECKPOINT_PATH}/label_enc_uid.pkl')
    label_enc_fid: preprocessing.LabelEncoder = joblib.load(f'{CHECKPOINT_PATH}/label_enc_fid.pkl')

    df_test.uid = label_enc_uid.transform(df_test.uid.values)
    df_test.fid = label_enc_fid.transform(df_test.fid.values)

    model = RecSysModel(
        num_users=len(label_enc_uid.classes_),
        num_items=len(label_enc_fid.classes_),
        layers=layers,
        reg_layers=reg_layers
    ).to(device)

    model.load_state_dict(
        torch.load(f'{CHECKPOINT_PATH}/ncf.pth', weights_only=True),
    )
    model.eval()

    test_dataset = MyDataset(
        list(df_test.uid),
        list(df_test.fid),
        list(df_test.rating)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    y_pred = []
    y_true = []
    user_ratings_comparison = defaultdict(list)
    with torch.no_grad():
        for i, test_data in enumerate(tqdm(test_loader)):
            uid = test_data['uid'].to(device)
            fid = test_data['fid'].to(device)
            rating = test_data["rating"].to(device)

            output = model(
                uid, fid
            )
            output = output.squeeze()

            y_pred.extend(output.cpu().numpy())
            y_true.extend(rating.cpu().numpy())
            for user, pred, true in zip(uid, output, rating):
                user_ratings_comparison[user.item()].append((pred.item(), true.item()))

    rmse = root_mean_squared_error(y_true, y_pred)
    avg_precision = 0.0
    avg_recall = 0.0

    for user, user_ratings in user_ratings_comparison.items():
        precision, recall = calculate_precision_recall(user_ratings, k, threshold)
        avg_precision += precision
        avg_recall += recall
    
    avg_precision /= len(user_ratings_comparison)
    avg_recall /= len(user_ratings_comparison)
    f1_score = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall)
    print(f'RMSE: {rmse:.4f}, precision: {avg_precision:.4f}, recall: {avg_recall:.4f}, f1:{f1_score:.4f}')

# RMSE: 1.5770, precision: 0.9035, recall: 0.7080, f1:0.7939

if __name__ == '__main__':
    eval()