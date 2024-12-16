from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Movie(db.Model):
    __tablename__ = 'movies'

    fid = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.String(500))
    ratingCount = db.Column(db.Integer)
    ratingValue = db.Column(db.Float)
    contentRating = db.Column(db.String(50))
    genre = db.Column(db.String(100))
    keywords = db.Column(db.String(200))
    duration = db.Column(db.String(50))
    datePublished = db.Column(db.String(50))
    actor = db.Column(db.String(200))
    director = db.Column(db.String(200))
    image = db.Column(db.String(200))

class Rating(db.Model):
    __tablename__ = 'ratings'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.String(100), nullable=False)
    film_ids = db.Column(db.Integer, db.ForeignKey('movies.fid'), nullable=False)
    rating = db.Column(db.Integer, nullable=False)

    movie = db.relationship('Movie', backref=db.backref('ratings', lazy=True))
