from typing import Literal

import numpy as np

from recommender.config import RATINGS_TO_PREDICT
from ..database import queries


class SGDFactorization:
    default_params = {'epochs': 70, 'n_factors': 80, 'learning_rate': 0.01, 'reg': 0.1}

    epochs = None
    n_factors = None
    learning_rate = None
    reg = None

    ratings_matrix = None
    n_users = None
    n_movies = None
    user_index_to_id = None
    movie_index_to_id = None
    non_zero_elems_row_ids = None
    non_zero_elems_col_ids = None
    training_indices = None

    user_bias = None
    movie_bias = None
    global_bias = None
    user_embeddings = None
    movie_embeddings = None

    current_epoch = 0
    predicted_ratings = None
    n_relevant_ratings_to_save = 50

    params = ['epochs', 'n_factors', 'learning_rate', 'reg']

    def __init__(self, params=None):
        if params is None:
            params = self.default_params

        for param, value in params.items():
            if param not in self.params:
                raise Exception(f'Unknown parameter {param}.')
            setattr(self, param, value)

    def pre_train_init(self):
        self.ratings_matrix, self.user_index_to_id, self.movie_index_to_id = queries.get_ratings_data()
        self.n_users, self.n_movies = len(self.user_index_to_id), len(self.movie_index_to_id)

        self.non_zero_elems_row_ids, self.non_zero_elems_col_ids = (~np.isnan(self.ratings_matrix)).nonzero()
        self.training_indices = np.arange(len(self.non_zero_elems_row_ids))

        self.global_bias = np.nanmean(self.ratings_matrix)
        self.user_bias = np.zeros(self.n_users)
        self.movie_bias = np.zeros(self.n_movies)

        self.user_embeddings = np.random.normal(
            scale=1. / self.n_factors,
            size=(self.n_users, self.n_factors),
        )
        self.movie_embeddings = np.random.normal(
            scale=1. / self.n_factors,
            size=(self.n_movies, self.n_factors),
        )

    def train(self):
        self.pre_train_init()

        while self.current_epoch < self.epochs:
            self.sgd()
            self.current_epoch += 1

        self.predicted_ratings = (
                self.global_bias + self.user_bias.reshape(-1, 1) + self.movie_bias.reshape(1, -1) +
                self.user_embeddings.dot(self.movie_embeddings.T)
        )

    def sgd(self):
        np.random.shuffle(self.training_indices)
        for idx in self.training_indices:
            user_idx = self.non_zero_elems_row_ids[idx]
            movie_idx = self.non_zero_elems_col_ids[idx]

            prediction = (
                    self.global_bias + self.user_bias[user_idx] + self.movie_bias[movie_idx] +
                    self.user_embeddings[user_idx].dot(self.movie_embeddings[movie_idx].T)
            )
            err = self.ratings_matrix[user_idx, movie_idx] - prediction

            user_embedding = self.user_embeddings[user_idx]
            movie_embedding = self.movie_embeddings[movie_idx]

            self.user_bias[user_idx] += self.learning_rate * (err - self.reg * self.user_bias[user_idx])
            self.movie_bias[movie_idx] += self.learning_rate * (err - self.reg * self.movie_bias[movie_idx])

            self.user_embeddings[user_idx] += self.learning_rate * (err * movie_embedding - self.reg * user_embedding)
            self.movie_embeddings[movie_idx] += self.learning_rate * (err * user_embedding - self.reg * movie_embedding)

    def save_object_embeddings(self, which: Literal['user', 'movie']):
        partial_global_bias = self.global_bias / 2
        if which not in ['user', 'movie']:
            raise Exception

        if which == 'user':
            n_objects = self.n_users
            object_bias = self.user_bias
            index_to_id = self.user_index_to_id
            object_embeddings = self.user_embeddings
        else:
            n_objects = self.n_movies
            object_bias = self.movie_bias
            index_to_id = self.movie_index_to_id
            object_embeddings = self.movie_embeddings

        object_embeds = []
        for object_index in range(n_objects):
            object_id = index_to_id[object_index]

            bias = partial_global_bias + object_bias[object_index]
            object_embeds.append((object_id, 0, bias))

            for embed_dim_no in range(self.n_factors):
                embed_value = object_embeddings[object_index][embed_dim_no]
                object_embeds.append((object_id, embed_dim_no + 1, embed_value))

        queries.save_object_embeddings(object_embeds, which=which)

    def save_predicted_ratings(self):
        movie_mapper_func = np.vectorize(lambda index: self.movie_index_to_id[index])
        self.predicted_ratings[(~np.isnan(self.ratings_matrix)).nonzero()] = np.nan

        predicted_ratings_sorted_order = (-1 * self.predicted_ratings).argsort(axis=1)
        sorted_predicted_ratings = np.take_along_axis(self.predicted_ratings, predicted_ratings_sorted_order, axis=1)
        sorted_movie_ids = movie_mapper_func(predicted_ratings_sorted_order)

        data = []
        for user_index in range(self.n_users):
            for i in range(RATINGS_TO_PREDICT):
                rating = sorted_predicted_ratings[user_index][i]
                if np.isnan(rating):
                    break

                user_id = self.user_index_to_id[user_index]
                movie_id = sorted_movie_ids[user_index][i]
                data.append((user_id, movie_id, rating))

        queries.save_predicted_ratings(data)
