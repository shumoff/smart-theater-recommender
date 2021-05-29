-- Create user memory embeddings table
-- depends: 03_recommender-create-ratings-table

CREATE TABLE user_memory_embeds
(
    user_id      integer REFERENCES user_id_mapping,
    embed_dim_no integer NOT NULL CHECK (embed_dim_no >= 0),
    embed_value real NOT NULL,
    CONSTRAINT pk_user_memory_embeds PRIMARY KEY (user_id, embed_dim_no)
);
