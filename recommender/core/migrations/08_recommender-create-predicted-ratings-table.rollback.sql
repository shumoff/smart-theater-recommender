-- Rollback predicted ratings table
-- depends: 07_recommender-create-movie-content-embeddings-table

DROP TABLE IF EXISTS predicted_ratings;