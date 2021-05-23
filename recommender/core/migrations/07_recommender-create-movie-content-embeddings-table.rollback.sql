-- Rollback movie content embeddings table
-- depends: 06_recommender-create-user-content-embeddings-table

DROP TABLE IF EXISTS movie_content_embeds;
