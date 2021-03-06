from common.movielens import MovieLens
from common.evaluator import Evaluator
from .rbm_wrapper import RBMAlgorithm
from surprise import NormalPredictor
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
    return ml, data, rankings


if __name__ == '__main__':
    # Load up common data set for the recommender algorithms
    ml, data, rankings = load_movielens()

    # Construct an Evaluator
    evaluator = Evaluator(data, rankings)
    evaluator.add_algorithm(RBMAlgorithm(epochs=20), "RBM")
    evaluator.add_algorithm(NormalPredictor(), "Random")

    evaluator.evaluate(True)
    evaluator.sample_topn_recs(ml)
