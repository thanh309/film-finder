from flask import Flask, request, session, redirect, url_for, render_template
import pandas as pd

rating_file = '../resources/combined_ratings.csv'
movie_file = '../test.csv'
db_users = pd.read_csv(rating_file) #user_id,film_ids,rating
db_movie = pd.read_csv(movie_file)  #fid,name,description,ratingCount,ratingValue,contentRating,genre,keywords,duration
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
        else:
            return render_template('login.html', error = "Invalid user")
    return render_template('login.html')

@app.route('/main', methods = ['GET', 'POST'])
def main():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    if 'user' == 'Guest':
        search_query = request.form.get('search', '').lower()
        if search_query:
            filtered_content = db_movie[db_movie['name'].str.contains(search_query, case = False, na = False)].to_dict('records')
        else:
            filtered_content = db_movie.to_dict('records') 
    else:
        search_query = request.form.get('search', '').lower()

        if search_query:
            filtered_content = db_movie[db_movie['name'].str.contains(search_query, case = False, na = False)].to_dict('records')
        else:
            filtered_content = db_movie.to_dict('records')

    return render_template('main.html', content = filtered_content)

@app.route('/guest', methods=['POST'])
def guest():
    # Assign a generic guest username
    session['user'] = 'Guest'
    return redirect(url_for('main'))

@app.route('/content/<int:content_id>')
def content(content_id):
    if 'user' not in session:
        return redirect(url_for('login'))
    
    item = db_movie[db_movie['fid'] == content_id].to_dict('records')

    if not item:
        return "Content no found", 404
    
    return render_template('content.html', item = item)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug = True)

