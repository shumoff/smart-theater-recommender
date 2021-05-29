-- Rollback movie index sequence
-- depends: 09_recommender-create-user-index-seq

DROP SEQUENCE IF EXISTS movie_index;
