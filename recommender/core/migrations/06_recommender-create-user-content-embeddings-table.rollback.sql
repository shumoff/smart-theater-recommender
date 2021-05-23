-- Rollback user content embeddings table
-- depends: 05_recommender-create-movie-memory-embeddings-table

DROP TABLE IF EXISTS user_content_embeds;
