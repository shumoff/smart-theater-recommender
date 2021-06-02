-- Create movie memory embeddings table
-- depends: 04_recommender-create-user-memory-embeddings-table

CREATE TABLE movie_memory_embeds
(
    movie_id     integer REFERENCES movie_id_mapping,
    embed_dim_no integer NOT NULL CHECK (embed_dim_no >= 0),
    embed_value double precision NOT NULL,
    CONSTRAINT pk_movie_memory_embeds PRIMARY KEY (movie_id, embed_dim_no)
);
