-- Create predicted similarities table
-- depends: 10_recommender-create-movie-index-seq

CREATE TABLE predicted_similarities
(
    movie_id         integer REFERENCES movie_id_mapping,
    similar_movie_id integer REFERENCES movie_id_mapping,
    similarity       double precision NOT NULL CHECK (similarity between -1 and 1),
    CONSTRAINT pk_predicted_similarities PRIMARY KEY (movie_id, similar_movie_id)
);
