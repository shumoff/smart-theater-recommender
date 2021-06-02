from typing import Literal

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from recommender.config import DB_NAME, DB_PASSWORD, DB_USER, DB_HOST, DB_PORT


def get_conn(another=False):
    if another:
        return psycopg2.connect(
            dbname='smart_theater_db',
            user='smart_theater_user',
            password='smart_theater_password',
            host='host.docker.internal',
            port=5432,
        )
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )


def get_ratings_data():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("select last_value from user_index;")
    users_amount = cur.fetchone()[0] + 1

    cur.execute("select last_value from movie_index;")
    movies_amount = cur.fetchone()[0] + 1

    ratings_matrix = np.full((users_amount, movies_amount), np.nan)

    ratings_query = """
    select uim.user_id, uim.user_index, mim.movie_id, mim.movie_index, r.rating
    from ratings r
             join user_id_mapping uim on r.user_id = uim.user_id
             join movie_id_mapping mim on r.movie_id = mim.movie_id
    where uim.user_index < %s and mim.movie_index < %s
    order by uim.user_index, mim.movie_index;
    """
    cur.execute(ratings_query, (users_amount, movies_amount))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    user_index_to_id = {}
    movie_index_to_id = {}
    for user_id, user_index, movie_id, movie_index, rating in rows:
        ratings_matrix[user_index][movie_index] = rating
        user_index_to_id[user_index] = user_id
        movie_index_to_id[movie_index] = movie_id

    return ratings_matrix, user_index_to_id, movie_index_to_id


def get_user_index_mapping():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("select user_index, user_id from user_id_mapping;")
    rows = cur.fetchall()
    user_index_to_id = dict(rows)
    cur.close()
    conn.close()

    return user_index_to_id


def get_movie_index_mapping():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("select movie_index, movie_id from movie_id_mapping;")
    rows = cur.fetchall()
    movie_index_to_id = dict(rows)
    cur.close()
    conn.close()

    return movie_index_to_id


def get_users_memory_embeds(user_ids):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        'select embed_dim_no, embed_value from user_memory_embeds where user_id in %s order by embed_dim_no;',
        (user_ids,)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    user_embeds = np.zeros((1, len(rows)))
    for _, embed_dim_no, value in rows:
        user_embeds[embed_dim_no] = value

    return user_embeds


def get_movies_memory_embeds(movie_ids):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        'select embed_dim_no, embed_value from movie_memory_embeds where movie_id = %s order by embed_dim_no;',
        (movie_ids,)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    movie_embeds = np.zeros((1, len(rows)))
    for _, embed_dim_no, value in rows:
        movie_embeds[embed_dim_no] = value

    return movie_embeds


def get_relevant_movies(user_id, offset, limit):
    conn = get_conn()
    cur = conn.cursor()

    query = """
    select pr.movie_id
    from predicted_ratings pr
            left join ratings r on pr.user_id = r.user_id and pr.movie_id = r.movie_id
    where pr.user_id = %s and r.rating is null
    order by pr.rating desc
    offset %s limit %s;
    """
    cur.execute(query, (user_id, offset, limit))
    rows = cur.fetchall()
    most_relevant_movies = [t[0] for t in rows]
    cur.close()
    conn.close()

    return most_relevant_movies


def get_similar_movies(movie_id, offset, limit, skip_watched=False):
    conn = get_conn()
    cur = conn.cursor()

    query = """
    select similar_movie_id from predicted_similarities
    where movie_id = %s
    order by similarity desc
    offset %s limit %s;
    """
    cur.execute(query, (movie_id, offset, limit))
    rows = cur.fetchall()
    similar_movies = [t[0] for t in rows]
    similar_movies = tuple(similar_movies)
    cur.close()
    conn.close()

    return similar_movies


def get_predicted_ratings_for_movies(movie_ids, offset, limit):
    conn = get_conn()
    cur = conn.cursor()

    # TODO RECOMMENDER-3: implement this logic
    query = """
    select rating from predicted_ratings
    where movie_id in %s
    order by rating desc
    offset %s limit %s;
    """
    cur.execute(query, (movie_ids,))
    rows = cur.fetchall()
    similar_movies = [t[0] for t in rows]
    cur.close()
    conn.close()

    return similar_movies


def save_new_rating(user_id, movie_id, rating):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        insert into ratings (user_id, movie_id, rating)
        values (%s, %s, %s)
        on conflict (user_id, movie_id) DO UPDATE SET rating = EXCLUDED.rating
        """,
        (user_id, movie_id, rating),
    )

    cur.execute(
        """
        delete from predicted_ratings
        where user_id = %s and movie_id = %s
        """,
        (user_id, movie_id)
    )

    cur.close()
    conn.commit()
    conn.close()


def save_predicted_ratings(data):
    conn = get_conn()
    cur = conn.cursor()

    execute_values(
        cur,
        """
        insert into predicted_ratings (user_id, movie_id, rating)
        values %s
        on conflict (user_id, movie_id) DO UPDATE SET rating = EXCLUDED.rating
        """,
        data,
        page_size=10000,
    )

    cur.close()
    conn.commit()
    conn.close()


def save_object_embeddings(data, which: Literal['user', 'movie']):
    if which not in ['user', 'movie']:
        raise Exception

    conn = get_conn()
    cur = conn.cursor()

    if which == 'user':
        execute_values(
            cur,
            """
            insert into user_memory_embeds (user_id, embed_dim_no, embed_value)
            values %s
            on conflict (user_id, embed_dim_no) DO UPDATE SET embed_value = EXCLUDED.embed_value
            """,
            data,
            page_size=10000,
        )
    else:
        execute_values(
            cur,
            """
            insert into movie_memory_embeds (movie_id, embed_dim_no, embed_value)
            values %s
            on conflict (movie_id, embed_dim_no) DO UPDATE SET embed_value = EXCLUDED.embed_value
            """,
            data,
            page_size=10000,
        )

    cur.close()
    conn.commit()
    conn.close()


def save_predicted_similarities(data):
    conn = get_conn()
    cur = conn.cursor()

    execute_values(
        cur,
        """
        insert into predicted_similarities (movie_id, similar_movie_id, similarity)
        values %s
        on conflict (movie_id, similar_movie_id) DO UPDATE SET similarity = EXCLUDED.similarity
        """,
        data,
        page_size=10000,
    )

    cur.close()
    conn.commit()
    conn.close()


def save_content_movie_embeddings(data):
    conn = get_conn()
    cur = conn.cursor()

    execute_values(
        cur,
        """
        insert into movie_content_embeds (movie_id, embed_dim_no, embed_value)
        values %s
        on conflict (movie_id, embed_dim_no) DO UPDATE SET embed_value = EXCLUDED.embed_value
        """,
        data,
        page_size=10000,
    )

    cur.close()
    conn.commit()
    conn.close()
