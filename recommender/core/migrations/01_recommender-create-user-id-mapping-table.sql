-- Create user id mapping table
-- depends:

CREATE TABLE user_id_mapping
(
    user_id    integer PRIMARY KEY,
    user_index integer UNIQUE NOT NULL,
    trained    boolean
);
