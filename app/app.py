from flask import Flask, request, session, redirect, url_for, render_template
import pandas as pd

rating_file = '../resources/combined_ratings.csv'
movie_file = '../test.csv'
db_users = pd.read_csv(rating_file) #user_id,film_ids,rating
db_movie = pd.read_csv(movie_file)  #fid,name,description,ratingCount,ratingValue,contentRating,genre,keywords,duration,datePublished,actor,director,image

users = db_users["user_id"].astype(str).unique()
db_movie['name'] = db_movie['name'].astype(str)
db_movie['fid'] = db_movie['fid'].astype(int)



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

    highest_rated_movies = db_movie.sort_values(by='ratingValue', ascending=False).head(10).to_dict('records')
    you_may_like_movies = [] if is_guest else db_movie.sample(10).to_dict('records')

    search_query = request.form.get('search', '').lower() if request.method == 'POST' else ''
    filtered_movies = db_movie[db_movie['name'].str.contains(search_query, case=False, na=False)].to_dict('records') if search_query else []

    return render_template(
        'main.html',
        user_name=user_name,
        is_guest=is_guest,
        highest_rated_movies=highest_rated_movies,
        you_may_like_movies=you_may_like_movies,
        search_results=filtered_movies
    )

@app.route('/content/<int:content_id>')
def content(content_id):
    item = db_movie[db_movie['fid'] == content_id].to_dict('records')
    if not item:
        return "Movie not found", 404
    return render_template('content.html', item=item[0])


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug = True)

