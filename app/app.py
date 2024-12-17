from flask import Flask, request, session, redirect, url_for, render_template
from sqlalchemy.exc import SQLAlchemyError
from model.model import db, Movie, Rating
# from content_based.knn_model import predict_film_unwatch
from ncf.ncf import recommend_top_movies



app = Flask(__name__)
app.config['SECRET_KEY'] = 'filmfinder'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///D:/others/film-finder/app/movies_ratings.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

with app.app_context():
    users = db.session.query(Rating.user_id).distinct().all()
    users = [int(u[0]) for u in users]

@app.route('/', methods = ['GET', 'POST'])
def login():
    if request.method == 'POST':    
        username = request.form['username']
        try:
            username_int = int(username)
        except ValueError:
            username_int = None
        # password = request.form['password']
        if username_int in users:
            session['user'] = username_int
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

    highest_rated_movies = Movie.query.order_by(Movie.ratingValue.desc()).limit(30).all()
    # you_may_like_movies = [] if is_guest else Movie.query.order_by(Movie.ratingValue.asc()).limit(30).all()
    you_may_like_movies = []
    if not is_guest:
        # recommended_fids = predict_film_unwatch(int(user_name))
        recommended_fids = recommend_top_movies(int(user_name))
        if isinstance(recommended_fids, list):
            recommended_fids = [int(fid) for fid in recommended_fids]
        you_may_like_movies = Movie.query.filter(Movie.fid.in_(recommended_fids)).all()

    search_query = request.form.get('search', '').lower() if request.method == 'POST' else ''
    filtered_movies = []

    if search_query:
        filtered_movies = Movie.query.filter(Movie.name.ilike(f" %{search_query}% ")).all()

    # highest_rated_movies = db_movie.sort_values(by='ratingValue', ascending=False).to_dict('records')
    # you_may_like_movies = [] if is_guest else db_movie.sample(10).sort_values(by='ratingValue', ascending = False).to_dict('records')

    # search_query = request.form.get('search', '').lower() if request.method == 'POST' else ''
    # filtered_movies = db_movie[db_movie['name'].str.contains(search_query, case=False, na=False)].to_dict('records') if search_query else []

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

@app.route('/update_you_may_like')
def update_you_may_like():
    user_id = session.get('user', 'guest')
    offset = int(request.args.get('offset', 0))
    limit = 5  

    recommended_fids = recommend_top_movies(user_id) 
    if isinstance(recommended_fids, list):
        recommended_fids = [int(fid) for fid in recommended_fids]
    you_may_like_movies = (
        Movie.query.filter(Movie.fid.in_(recommended_fids))
        .offset(offset)
        .limit(limit)
        .all()
    )
    movies = [{'fid': movie.fid, 'name': movie.name, 'image': movie.image} for movie in you_may_like_movies]
    return {'movies': movies}


@app.route('/update_highest_rated')
def update_highest_rated():
    offset = int(request.args.get('offset', 0))
    highest_rated_movies = Movie.query.order_by(Movie.ratingValue.desc()).offset(offset).limit(5).all()
    movies = [{'fid': movie.fid, 'name': movie.name, 'image': movie.image} for movie in highest_rated_movies]
    return {'movies': movies}


@app.route('/content/<int:content_id>', methods=['GET', 'POST'])
def content(content_id):
    # movie = db_movie[db_movie['fid'] == content_id].to_dict('records')
    movie = Movie.query.filter_by(fid = content_id).first()

    if not movie:
        return "Movie not found", 404

    # movie = movie[0]

    user_id = session.get('user', 'guest')
    is_guest = 'guest' in session

    if not is_guest:
        # user_rating = db_users[(db_users['user_id'] == user_id) & (db_users['film_ids'] == content_id)]
        # user_rating = user_rating['rating'].iloc[0] if not user_rating.empty else None
        user_rating = Rating.query.filter_by(user_id = user_id, film_ids = content_id).first()

    if request.method == 'POST' and not is_guest:
        data = request.get_json()
        new_rating = data.get('rating')
        
        # if new_rating and isinstance(new_rating, int) and 1 <= new_rating <= 10:
        #     if user_rating is None:
        #         db_users.loc[len(db_users)] = {'user_id': user_id, 'film_ids': content_id, 'rating': new_rating}
        #         save_new_rating(user_id, content_id, new_rating)
        #     else:
        #         db_users.loc[(db_users['user_id'] == user_id) & (db_users['film_ids'] == content_id), 'rating'] = new_rating
        #         save_new_rating(user_id, content_id, new_rating)
        #     return {'success': True}, 200
        # else:
        #     return {'success': False, 'error': 'Invalid rating'}, 400

        if new_rating and isinstance(new_rating, int) and 1 <= new_rating <= 10:
            try:
                if user_rating:
                    user_rating.rating = new_rating
                else:
                    new_rating_entry = Rating(user_id = user_id, film_ids = content_id, rating = new_rating)
                    db.session.add(new_rating_entry)
                
                db.session.commit()
                return {'success': True}, 200
            
            except SQLAlchemyError as e:
                db.session.rollback()
                return{'success': False, 'error': str(e)}, 500
        else:
            return {'success': False, 'error': 'Invalid rating'}, 400

    return render_template(
        'content.html',
        movie=movie,
        user_rating=user_rating.rating if user_rating else None,
        is_guest=is_guest
    )
@app.route('/rated_movies', methods=['GET'])
def rated_movies():
    user_id = session.get('user', None)
    
    rated_movies = (
        db.session.query(Movie)
        .join(Rating, Movie.fid == Rating.film_ids)
        .filter(Rating.user_id == user_id)
        .all()
    )
    return render_template('rated_movies.html', rated_movies=rated_movies)


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug = True)



