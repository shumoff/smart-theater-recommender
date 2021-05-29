-- Rollback user index sequence
-- depends: 08_recommender-create-predicted-ratings-table

DROP SEQUENCE IF EXISTS user_index;
