from concurrent import futures
import random

import grpc

from recommender_pb2 import MovieRecommendation, RecommendationResponse
import recommender_pb2_grpc


movies = [
    MovieRecommendation(id=1, title="The Maltese Falcon"),
    MovieRecommendation(id=2, title="Murder on the Orient Express"),
    MovieRecommendation(id=3, title="The Hound of the Baskervilles"),
    MovieRecommendation(id=4, title="The Hitchhiker's Guide to the Galaxy"),
    MovieRecommendation(id=5, title="Ender's Game"),
    MovieRecommendation(id=6, title="The Dune Chronicles"),
    MovieRecommendation(id=7, title="The 7 Habits of Highly Effective People"),
    MovieRecommendation(id=8, title="How to Win Friends and Influence People"),
    MovieRecommendation(id=9, title="Man's Search for Meaning"),
]


class RecommendationService(recommender_pb2_grpc.RecommenderServicer):
    def Recommend(self, request, context):
        # if request.category not in books_by_category:
        #     context.abort(grpc.StatusCode.NOT_FOUND, "Category not found")

        num_results = min(request.max_results, len(movies))
        recommended_movies = random.sample(movies, num_results)

        return RecommendationResponse(recommendations=recommended_movies)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    recommender_pb2_grpc.add_RecommenderServicer_to_server(
        RecommendationService(), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
