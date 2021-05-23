-- Rollback movie id mapping table
-- depends: 01_recommender-create-user-id-mapping-table

DROP TABLE IF EXISTS movie_id_mapping;
