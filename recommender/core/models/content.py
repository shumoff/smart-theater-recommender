import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model


from recommender.config import SIMILAR_MOVIES_TO_SAVE
from recommender.core.database import queries


class Autoencoder(Model):
    def __init__(self, input_dim, intermediate_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.BatchNormalization(),
            layers.Dense(intermediate_dim, activation='relu'),
            layers.Dropout(0.2, seed=42),
            layers.BatchNormalization(),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.BatchNormalization(),
            layers.Dense(intermediate_dim, activation='relu'),
            layers.Dropout(0.2, seed=42),
            layers.BatchNormalization(),
            layers.Dense(input_dim, activation='sigmoid'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ContentFiltering:
    default_params = {'intermediate_size': 5000, 'encoded_size': 100, 'epochs': 20, 'use_tfidf': False}

    epochs = None
    intermediate_size = None
    encoded_size = None
    use_tfidf = None

    n_movies = None
    movie_index_to_id = None

    movie_embeddings = None

    current_epoch = 0

    embeddings_matrix = None

    params = ['intermediate_size', 'encoded_size', 'epochs', 'use_tfidf']

    def __init__(self, movie_content_data_path, params=None):
        self.movie_content_data = pd.read_csv(movie_content_data_path)

        if params is None:
            params = self.default_params

        for param, value in params.items():
            if param not in self.params:
                raise Exception(f'Unknown parameter {param}.')
            setattr(self, param, value)

    def pre_train_init(self):
        self.movie_index_to_id = queries.get_movie_index_mapping()
        self.n_movies = len(self.movie_index_to_id)
        return
        self.movie_content_data = self.movie_content_data[
            self.movie_content_data['movieId'].isin(self.movie_index_to_id.values())
        ]
        movie_id_to_index = {v: k for k, v in self.movie_index_to_id.items()}

        self.movie_content_data.replace({'movieId': movie_id_to_index}, inplace=True)
        self.movie_content_data.sort_values(by='movieId', ignore_index=True, inplace=True)

        if self.use_tfidf:
            tfidf_vectorizer = TfidfVectorizer(
                ngram_range=(1, 1),
                min_df=0.0001,
                stop_words='english',
                token_pattern=r"(?u)\b\w[\w\-]+\b",
            )
            self.embeddings_matrix = tfidf_vectorizer.fit_transform(self.movie_content_data['document']).toarray()
        else:
            tag_id_to_index = {tag_id: tag_ix for tag_ix, tag_id in enumerate(self.movie_content_data['tagId'].unique())}
            self.embeddings_matrix = np.zeros((self.n_movies, len(tag_id_to_index)))
            for _, movie_id, tag_id, relevance in self.movie_content_data.itertuples():
                self.embeddings_matrix[movie_id][tag_id_to_index[tag_id]] = relevance

    def train(self):
        self.pre_train_init()
        tf.random.set_seed(42)
        autoencoder = Autoencoder(self.embeddings_matrix.shape[1], self.intermediate_size, self.encoded_size)
        autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

        train_tfidf, test_tfidf = train_test_split(
            self.embeddings_matrix,
            test_size=0.2,
            random_state=42,
        )

        autoencoder.fit(
            train_tfidf, train_tfidf,
            epochs=self.epochs,
            shuffle=True,
            validation_data=(test_tfidf, test_tfidf)
        )

        self.movie_embeddings = autoencoder.encoder(self.embeddings_matrix).numpy()

    def save_movie_embeddings(self):
        # np.savetxt(os.path.join('/code', 'volumes', 'backend', 'embeddings.csv'), self.movie_embeddings, delimiter=',')
        self.movie_embeddings = np.genfromtxt(os.path.join('/code', 'volumes', 'backend', 'embeddings.csv'), delimiter=',')
        return
        object_embeds = []
        for movie_index in range(self.n_movies):
            movie_id = self.movie_index_to_id[movie_index]

            for embed_dim_no in range(self.encoded_size):
                embed_value = self.movie_embeddings[movie_index][embed_dim_no]
                object_embeds.append((movie_id, embed_dim_no, embed_value))

        queries.save_content_movie_embeddings(object_embeds)

    def save_similar_movies(self):
        similarity_matrix = cosine_similarity(self.movie_embeddings)

        movie_mapper_func = np.vectorize(lambda index: self.movie_index_to_id[index])

        similarities_sorted_order = (-1 * similarity_matrix).argsort(axis=1)
        sorted_similarities = np.take_along_axis(similarity_matrix, similarities_sorted_order, axis=1)
        sorted_movie_ids = movie_mapper_func(similarities_sorted_order)

        data = []
        for movie_index in range(self.n_movies):
            for similar_movie_index in range(1, SIMILAR_MOVIES_TO_SAVE + 1):
                similarity = min(sorted_similarities[movie_index][similar_movie_index], 1)

                movie_id = self.movie_index_to_id[movie_index]
                similar_movie_id = sorted_movie_ids[movie_index][similar_movie_index]
                data.append((movie_id, int(similar_movie_id), float(similarity)))

        queries.save_predicted_similarities(data)


if __name__ == '__main__':
    import os
    model = ContentFiltering(
        movie_content_data_path=os.path.join('/code', 'volumes', 'backend', 'movie_genome_scores.csv'),
        params=ContentFiltering.default_params,
    )
    model.train()
    model.save_movie_embeddings()
    model.save_similar_movies()
