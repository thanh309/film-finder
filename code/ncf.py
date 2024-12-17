import pandas as pd
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
import wandb
import datetime
from tqdm import tqdm
from collections import defaultdict
import joblib

class MyDataset(Dataset):
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def __len__(self):
        return len(self.users)

    def __getitem__(self, item):
        users = self.users[item]
        movies = self.movies[item]
        ratings = self.ratings[item]

        return {
            'uid': torch.tensor(users, dtype=torch.int),
            'fid': torch.tensor(movies, dtype=torch.int),
            'rating': torch.tensor(ratings, dtype=torch.float),
        }

class RecSysModel(nn.Module):
    def __init__(self, num_users, num_items, layers, reg_layers):
        super(RecSysModel, self).__init__()
        self.num_layers = len(layers)

        # split for concat later
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