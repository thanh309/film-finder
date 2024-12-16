from flask import Flask, request, session, redirect, url_for, render_template
import pandas as pd
import os

# rating_file = 'resources/combined_ratings.csv'
# movie_file = 'test.csv'
# db_users = pd.read_csv(rating_file) #user_id,film_ids,rating
# db_movie = pd.read_csv(movie_file)  #fid,name,description,ratingCount,ratingValue,contentRating,genre,keywords,duration,datePublished,actor,director,image

# users = db_users["user_id"].astype(str).unique()
# db_movie['name'] = db_movie['name'].astype(str)
# db_movie['fid'] = db_movie['fid'].astype(int)

movie_dir = "resources/data/split_film_data"
rating_dir = "resources/ratings"
rating_add = "resources/ratings/users_ratings_add.csv"

def load_movies():
    movie_files = [os.path.join(movie_dir, f) for f in os.listdir(movie_dir) if f.endswith('.csv')]
    movies = pd.concat([pd.read_csv(file) for file in movie_files], ignore_index=True)
    movies['fid'] = movies['fid'].astype(int)
    movies['name'] = movies['name'].astype(str)
    return movies

def load_ratings():
    rating_files = [os.path.join(rating_dir, f) for f in os.listdir(rating_dir) if f.endswith('.csv')]
    ratings = pd.concat([pd.read_csv(file, header=None, names=['user_id', 'film_ids', 'rating']) for file in rating_files], ignore_index=True)
    ratings['user_id'] = ratings['user_id'].astype(str)
    ratings['film_ids'] = ratings['film_ids'].astype(int)
    ratings['rating'] = ratings['rating'].astype(int)
    return ratings

def save_new_rating(user_id, film_id, rating):
    df = pd.read_csv(rating_add, names = ['user_id', 'film_ids', 'rating'])
    df['user_id'] = df['user_id'].astype(str)
    df['film_ids'] = df['film_ids'].astype(str)
    new_row = {'user_id': user_id, 'film_ids': film_id, 'rating': rating}
    condition = (df['user_id'] == str(user_id)) & (df['film_ids'] == str(film_id))
    if condition.any():
        df.loc[condition, 'rating'] = rating
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df = df.drop_duplicates(subset=['user_id', 'film_ids'], keep='last')

    df.to_csv(rating_add, mode = 'w', header = False, index=False)

db_movie = load_movies()
db_users = load_ratings()
users = db_users['user_id'].unique()

app = Flask(__name__)
app.secret_key = 'filmfinder'

@app.route('/', methods = ['GET', 'POST'])
def login():
    if request.method == 'POST':    
        username = request.form['username']
        # password = request.form['password']
        if username in users:
            session['user'] = username
            return redirect(url_for('main'))
        elif username.lower() == 'guest':
            session['user'] = 'guest'
            return redirect(url_for('main'))
        else:
            return render_template('login.html', error = "Invalid user")
    return render_template('login.html')

@app.route('/main', methods = ['GET', 'POST'])
def main():
    is_guest = session.get('user') == 'guest'
    user_name = session.get('user','guest')
    display_index_you_may_like = int(request.args.get('display_index_you_may_like', 0))
    display_index_highest_rated = int(request.args.get('display_index_highest_rated', 0))

    highest_rated_movies = db_movie.sort_values(by='ratingValue', ascending=False).to_dict('records')
    you_may_like_movies = [] if is_guest else db_movie.sample(10).sort_values(by='ratingValue', ascending = False).to_dict('records')

    search_query = request.form.get('search', '').lower() if request.method == 'POST' else ''
    filtered_movies = db_movie[db_movie['name'].str.contains(search_query, case=False, na=False)].to_dict('records') if search_query else []

    return render_template(
        'main.html',
        user_name=user_name,
        is_guest=is_guest,
        highest_rated_movies=highest_rated_movies,
        you_may_like_movies=you_may_like_movies,
        search_results=filtered_movies,
        display_index_you_may_like=display_index_you_may_like,
        display_index_highest_rated=display_index_highest_rated
    )

@app.route('/content/<int:content_id>', methods=['GET', 'POST'])
def content(content_id):
    movie = db_movie[db_movie['fid'] == content_id].to_dict('records')
    if not movie:
        return "Movie not found", 404

    movie = movie[0]

    user_id = session.get('user', 'guest')
    is_guest = 'guest' in session

    if not is_guest:
        user_rating = db_users[(db_users['user_id'] == user_id) & (db_users['film_ids'] == content_id)]
        user_rating = user_rating['rating'].iloc[0] if not user_rating.empty else None

    if request.method == 'POST' and not is_guest:
        data = request.get_json()
        new_rating = data.get('rating')
        if new_rating and isinstance(new_rating, int) and 1 <= new_rating <= 10:
            if user_rating is None:
                db_users.loc[len(db_users)] = {'user_id': user_id, 'film_ids': content_id, 'rating': new_rating}
                save_new_rating(user_id, content_id, new_rating)
            else:
                db_users.loc[(db_users['user_id'] == user_id) & (db_users['film_ids'] == content_id), 'rating'] = new_rating
                save_new_rating(user_id, content_id, new_rating)
            return {'success': True}, 200
        else:
            return {'success': False, 'error': 'Invalid rating'}, 400

    return render_template(
        'content.html',
        movie=movie,
        user_rating=user_rating,
        is_guest=is_guest
    )



@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug = True)

