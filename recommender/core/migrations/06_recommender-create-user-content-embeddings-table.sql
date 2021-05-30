-- Create user content embeddings table
-- depends: 05_recommender-create-movie-memory-embeddings-table

CREATE TABLE user_content_embeds
(
    user_id      integer REFERENCES user_id_mapping,
    embed_dim_no integer NOT NULL CHECK (embed_dim_no >= 0),
    embed_value real NOT NULL,
    CONSTRAINT pk_user_content_embeds PRIMARY KEY (user_id, embed_dim_no)
);
