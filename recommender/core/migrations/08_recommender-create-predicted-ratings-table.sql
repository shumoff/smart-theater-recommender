-- Create predicted ratings table
-- depends: 07_recommender-create-movie-content-embeddings-table

CREATE TABLE predicted_ratings
(
    user_id  integer REFERENCES user_id_mapping,
    movie_id integer REFERENCES movie_id_mapping,
    rating   double precision NOT NULL
);
