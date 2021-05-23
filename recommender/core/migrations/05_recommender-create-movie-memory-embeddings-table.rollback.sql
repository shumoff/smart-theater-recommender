-- Rollback movie memory embeddings table
-- depends: 04_recommender-create-user-memory-embeddings-table

DROP TABLE IF EXISTS movie_memory_embeds;
