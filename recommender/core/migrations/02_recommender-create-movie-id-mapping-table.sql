-- Rollback movie id mapping table
-- depends: 01_recommender-create-user-id-mapping-table

CREATE TABLE movie_id_mapping
(
    movie_id    integer PRIMARY KEY,
    movie_index integer UNIQUE NOT NULL,
    trained     boolean        NOT NULL
);
