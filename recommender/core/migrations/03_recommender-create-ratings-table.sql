-- Create ratings table
-- depends: 02_recommender-create-movie-id-mapping-table

CREATE TABLE ratings
(
    user_id  integer REFERENCES user_id_mapping,
    movie_id integer REFERENCES movie_id_mapping,
    rating   integer NOT NULL CHECK (rating > 0),
    CONSTRAINT pk_ratings PRIMARY KEY (user_id, movie_id)
);
