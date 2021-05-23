-- Rollback ratings table
-- depends: 02_recommender-create-movie-id-mapping-table

DROP TABLE IF EXISTS ratings;
