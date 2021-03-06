from concurrent import futures

import grpc

from recommender.core.database import queries

from .recommender_pb2 import EmptyResponse, MovieRecommendation, RecommendationResponse
from . import recommender_pb2_grpc

MAX_OFFSET = 100
MAX_LIMIT = 100


class RecommendationService(recommender_pb2_grpc.RecommenderServicer):
    def RecommendMovie(self, request, context):
        # if request.category not in books_by_category:
        #     context.abort(grpc.StatusCode.NOT_FOUND, "Category not found")

        offset = min(request.offset, MAX_OFFSET)
        limit = min(request.limit, MAX_LIMIT)

        recommended_movies_ids = queries.get_relevant_movies(request.user_id, offset, limit)
        recommended_movies = [MovieRecommendation(id=movie_id) for movie_id in recommended_movies_ids]

        return RecommendationResponse(recommendations=recommended_movies)

    def RecommendSimilarMovie(self, request, context):
        offset = min(request.offset, MAX_OFFSET)
        limit = min(request.limit, MAX_LIMIT)

        similar_movies_ids = queries.get_similar_movies(request.movie_id, offset, limit)
        similar_movies = [MovieRecommendation(id=movie_id) for movie_id in similar_movies_ids]

        return RecommendationResponse(recommendations=similar_movies)

    def RecommendRelevantSimilarMovie(self, request, context):
        offset = min(request.offset, MAX_OFFSET)
        limit = min(request.limit, MAX_LIMIT)

        # TODO RECOMMENDER-3: reinvent offsets
        similarity_offset = limit * (offset // limit)
        relevant_offset = offset % limit
        similar_movies_ids = queries.get_similar_movies(request.movie_id, similarity_offset, limit * 2)
        relevant_similar_movies_ids = queries.get_predicted_ratings_for_movies(similar_movies_ids, relevant_offset, limit)
        relevant_similar_movies = [MovieRecommendation(id=movie_id) for movie_id in relevant_similar_movies_ids]

        return RecommendationResponse(recommendations=relevant_similar_movies)

    def SaveNewRating(self, request, context):
        queries.save_new_rating(request.user_id, request.movie_id, request.rating)

        return EmptyResponse(ok=True)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    recommender_pb2_grpc.add_RecommenderServicer_to_server(
        RecommendationService(), server,
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
