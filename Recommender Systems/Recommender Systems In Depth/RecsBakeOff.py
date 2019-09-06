from common.movielens import MovieLens
from surprise import SVD, NormalPredictor
from common.evaluator import Evaluator
import random
import numpy as np

np.random.seed(0)
random.seed(0)


def load_movielens():
    ml = MovieLens()
    print("Loading movie ratings...")
    data = ml.load()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.get_popularity_ranks()
    return data, rankings


if __name__ == '__main__':
    # Load up common data set for the recommender algorithms
    evaluation_data, rankings = load_movielens()

    # Construct an Evaluator
    evaluator = Evaluator(evaluation_data, rankings)

    # Throw in an SVD recommender
    evaluator.add_algorithm(SVD(random_state=10), "SVD")

    # Just make random recommendations
    evaluator.add_algorithm(NormalPredictor(), "Random")
    evaluator.evaluate(True)

