import sqlite3
import pandas as pd
import os

# Directories
movie_dir = "resources/data/split_film_data"
rating_dir = "resources/ratings"

def load_movies():
    movie_files = [os.path.join(movie_dir, f) for f in os.listdir(movie_dir) if f.endswith('.csv')]
    movies = pd.concat([pd.read_csv(file) for file in movie_files], ignore_index=True)
    movies['fid'] = movies['fid'].astype(int)
    movies['name'] = movies['name'].astype(str)
    movies['description'] = movies['description'].fillna('').astype(str)
    movies['ratingCount'] = movies['ratingCount'].fillna(0).astype(int)
    movies['ratingValue'] = movies['ratingValue'].fillna(0).astype(float)
    movies['contentRating'] = movies['contentRating'].fillna('').astype(str)
    movies['genre'] = movies['genre'].fillna('').astype(str)
    movies['keywords'] = movies['keywords'].fillna('').astype(str)
    movies['duration'] = movies['duration'].fillna(0).astype(int)
    movies['datePublished'] = movies['datePublished'].fillna('').astype(str)
    movies['actor'] = movies['actor'].fillna('').astype(str)
    movies['director'] = movies['director'].fillna('').astype(str)
    movies['image'] = movies['image'].fillna('').astype(str)
    return movies

def load_ratings():
    rating_files = [os.path.join(rating_dir, f) for f in os.listdir(rating_dir) if f.endswith('.csv')]
    ratings = pd.concat([pd.read_csv(file, header=None, names=['user_id', 'film_ids', 'rating']) for file in rating_files], ignore_index=True)
    ratings['user_id'] = ratings['user_id'].astype(str)
    ratings['film_ids'] = ratings['film_ids'].astype(int)
    ratings['rating'] = ratings['rating'].astype(int)
    return ratings

conn = sqlite3.connect("app/movies_ratings.db")
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS movies")
cursor.execute("DROP TABLE IF EXISTS ratings")

cursor.execute('''CREATE TABLE IF NOT EXISTS movies (
                    fid INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    ratingCount INTEGER DEFAULT 0,
                    ratingValue REAL DEFAULT 0,
                    contentRating TEXT,
                    genre TEXT,
                    keywords TEXT,
                    duration INTEGER DEFAULT 0,
                    datePublished TEXT,
                    actor TEXT,
                    director TEXT,
                    image TEXT
                )''')

cursor.execute('''CREATE TABLE IF NOT EXISTS ratings (
                    user_id TEXT,
                    film_ids INTEGER,
                    rating INTEGER NOT NULL,
                    PRIMARY KEY (user_id, film_ids),
                    FOREIGN KEY (film_ids) REFERENCES movies(fid)
                )''')

movies = load_movies()
ratings = load_ratings()

for _, movie in movies.iterrows():
    cursor.execute('''
        INSERT OR REPLACE INTO movies (fid, name, description, ratingCount, ratingValue, contentRating, genre, keywords, 
                                      duration, datePublished, actor, director, image) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (movie['fid'], movie['name'], movie['description'], movie['ratingCount'], movie['ratingValue'], movie['contentRating'], 
          movie['genre'], movie['keywords'], movie['duration'], movie['datePublished'], movie['actor'], movie['director'], movie['image']))

for _, rating in ratings.iterrows():
    cursor.execute("INSERT OR REPLACE INTO ratings (user_id, film_ids, rating) VALUES (?, ?, ?)",
                   (rating['user_id'], rating['film_ids'], rating['rating']))

conn.commit()
conn.close()

print("Data loaded into SQLite successfully!")
