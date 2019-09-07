from common.movielens import MovieLens
from common.evaluator import Evaluator
from surprise import SVD, NormalPredictor
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
    param_grid = {'n_epochs': [20, 30], 'lr_all': [0.005, 0.010], 'n_factors': [50, 100]}
    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
    gs.fit(data)

    # best RMSE score
    print("Best RMSE score attained: ", gs.best_score['rmse'])

    # combination of parameters that gave the best RMSE score
    print(gs.best_params['rmse'])

    # Construct an Evaluator
    evaluator = Evaluator(data, rankings)

    params = gs.best_params['rmse']
    SVD_tuned = SVD(n_epochs=params['n_epochs'], lr_all=params['lr_all'], n_factors=params['n_factors'])
    evaluator.add_algorithm(SVD_tuned, "SVD - Tuned")

    SVD_untuned = SVD()
    evaluator.add_algorithm(SVD_untuned, "SVD - Untuned")

    # Just make random recommendations
    evaluator.add_algorithm(NormalPredictor(), "Random")
    evaluator.evaluate(False)
    evaluator.sample_topn_recs(ml)
