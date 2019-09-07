from common.movielens import MovieLens
from common.evaluator import Evaluator
from .rbm_wrapper import RBMAlgorithm
from surprise import NormalPredictor
from surprise.model_selection import GridSearchCV
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

    print("Searching for best parameters...")
    param_grid = {'hidden_dim': [20, 10], 'learning_rate': [0.1, 0.01]}
    gs = GridSearchCV(RBMAlgorithm, param_grid, measures=['rmse', 'mae'], cv=3)
    gs.fit(data)

    # best RMSE score
    print("Best RMSE score attained: ", gs.best_score['rmse'])

    # combination of parameters that gave the best RMSE score
    print(gs.best_params['rmse'])

    # Construct an Evaluator
    evaluator = Evaluator(data, rankings)
    params = gs.best_params['rmse']

    RBM_tuned = RBMAlgorithm(hidden_dim = params['hidden_dim'], learning_rate = params['learning_rate'])
    evaluator.add_algorithm(RBM_tuned, "RBM - Tuned")

    RBM_untuned = RBMAlgorithm()
    evaluator.add_algorithm(RBM_untuned, "RBM - Untuned")

    # Just make random recommendations
    evaluator.add_algorithm(NormalPredictor(), "Random")

    evaluator.evaluate(False)
    evaluator.sample_topn_recs(ml)
