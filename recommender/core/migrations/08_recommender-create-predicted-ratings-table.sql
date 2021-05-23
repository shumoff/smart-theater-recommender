-- Create predicted ratings table
-- depends: 07_recommender-create-movie-content-embeddings-table

CREATE TABLE predicted_ratings
(
    user_id  integer REFERENCES user_id_mapping,
    movie_id integer REFERENCES movie_id_mapping,
    rating   integer NOT NULL CHECK (rating > 0),
    CONSTRAINT pk_predicted_ratings PRIMARY KEY (user_id, movie_id)
);
