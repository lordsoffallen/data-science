from common.movielens import MovieLens
from surprise import KNNBasic
from heapq import nlargest
from collections import defaultdict
from operator import itemgetter


test_subject = '85'
k = 10   # Top k suggestion

if __name__ == '__main__':
    # Load our data set and compute the user similarity matrix
    ml = MovieLens()
    data = ml.load()

    trainset = data.build_full_trainset()
    sim_options = {'name': 'cosine', 'user_based': False}

    model = KNNBasic(sim_options=sim_options)
    model.fit(trainset)
    sims_matrix = model.compute_similarities()

    test_user_inner_id = trainset.to_inner_uid(test_subject)

    # Get the top K items we rated
    ratings = trainset.ur[test_user_inner_id]
    k_neighbors = nlargest(k, ratings, key=lambda t: t[1])

    # Alternate approach would be to select items up to some similarity threshold
    # k_neighbors = [rating for rating in ratings if rating[1] > 4.0]

    # Get similar items to stuff we liked (weighted by rating)
    candidates = defaultdict(float)
    for item_id, rating in k_neighbors:
        similarity_row = sims_matrix[item_id]
        for inner_id, score in enumerate(similarity_row):
            candidates[inner_id] += score * (rating / 5.0)

    # Build a dictionary of stuff the user has already seen
    watched = {item_id:1 for item_id, rating in trainset.ur[test_user_inner_id]}

    # Get top-rated items from similar users
    pos = 0
    for item_id, rating_sum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
        if item_id not in watched:
            movie_id = trainset.to_raw_iid(item_id)
            print(ml.get_movie_name(int(movie_id)), rating_sum)
            pos += 1
            if pos > 10: break
