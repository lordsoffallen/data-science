from common.movielens import MovieLens
from common.evaluator import Evaluator
from surprise import KNNBasic
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

    # Construct an Evaluator to, you know, evaluate them
    evaluator = Evaluator(data, rankings)

    # User-based KNN
    user_knn = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
    evaluator.add_algorithm(user_knn, "User KNN")

    # Item-based KNN
    item_knn = KNNBasic(sim_options={'name': 'cosine', 'user_based': False})
    evaluator.add_algorithm(item_knn, "Item KNN")

    # Just make random recommendations
    evaluator.add_algorithm(NormalPredictor(), "Random")
    evaluator.evaluate(False)
    evaluator.sample_topn_recs(ml)
