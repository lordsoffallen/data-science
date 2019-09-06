from common.movielens import MovieLens
from common.metrics import RecommenderMetrics as metrics
from common.data import EvaluationData
from surprise import KNNBasic
from heapq import nlargest
from collections import defaultdict
from operator import itemgetter


def load_movielens():
    ml = MovieLens()
    print("Loading movie ratings...")
    data = ml.load()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.get_popularity_ranks()
    return ml, data, rankings


if __name__ == '__main__':
    ml, data, rankings = load_movielens()
    evaluator = EvaluationData(data, rankings)

    # Train on leave-One-Out train set
    trainset = evaluator.get_loocv_trainset()
    sim_options = {'name': 'cosine', 'user_based': True}
    model = KNNBasic(sim_options=sim_options)
    model.fit(trainset)
    sims_matrix = model.compute_similarities()

    leftout_testset = evaluator.get_loocv_testset()

    # Build up dict to lists of (int(movieID), predictedrating) pairs
    top_n = defaultdict(list)
    k = 10
    for uiid in range(trainset.n_users):
        # Get top N similar users to this one
        similarity_row = sims_matrix[uiid]
        similar_users = [(inner_id, score) for inner_id, score in enumerate(similarity_row) if inner_id != uiid]
        k_neighbors = nlargest(k, similar_users, key=lambda t: t[1])

        # Get the stuff they rated, and add up ratings for each item, weighted by user similarity
        candidates = defaultdict(float)
        for user in k_neighbors:
            inner_id, user_similarity = user[0], user[1]
            ratings = trainset.ur[inner_id]
            for rating in ratings:
                candidates[rating[0]] += (rating[1] / 5.0) * user_similarity

        # Build a dictionary of stuff the user has already seen
        watched = {item_id: 1 for item_id, rating in trainset.ur[uiid]}

        # Get top-rated items from similar users
        pos = 0
        for item_id, rating_sum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
            if item_id not in watched:
                movie_id = trainset.to_raw_iid(item_id)
                top_n[int(trainset.to_raw_uid(uiid))].append((int(movie_id), 0.0))
                pos += 1
                if pos > 40: break

    print("HR", metrics.hit_rate(top_n, leftout_testset))


