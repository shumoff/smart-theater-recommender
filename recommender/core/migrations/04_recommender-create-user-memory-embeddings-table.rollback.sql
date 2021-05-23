-- Rollback user memory embeddings table
-- depends: 03_recommender-create-ratings-table

DROP TABLE IF EXISTS user_memory_embeds;
