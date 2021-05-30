-- Create movie content embeddings table
-- depends: 06_recommender-create-user-content-embeddings-table

CREATE TABLE movie_content_embeds
(
    movie_id     integer REFERENCES movie_id_mapping,
    embed_dim_no integer NOT NULL CHECK (embed_dim_no >= 0),
    embed_value real NOT NULL,
    CONSTRAINT pk_movie_content_embeds PRIMARY KEY (movie_id, embed_dim_no)
);
