from common.movielens import MovieLens
from surprise import KNNBasic
from collections import defaultdict
from operator import itemgetter
from heapq import nlargest


test_subject = '85'
k = 10   # Top k suggestion

if __name__ == '__main__':
    # Load our data set and compute the user similarity matrix
    ml = MovieLens()
    data = ml.load()

    trainset = data.build_full_trainset()
    sim_options = {'name': 'cosine', 'user_based': True}

    model = KNNBasic(sim_options=sim_options)
    model.fit(trainset)
    sims_matrix = model.compute_similarities()

    # Get top N similar users to our test subject
    test_user_inner_id = trainset.to_inner_uid(test_subject)
    similarity_row = sims_matrix[test_user_inner_id]

    similar_users = [(inner_id, score) for inner_id, score in enumerate(similarity_row)
                     if inner_id != test_user_inner_id]

    k_neighbors = nlargest(k, similar_users, key=lambda t: t[1])

    # Alternate approach would be to select users up to some similarity threshold
    # k_neighbors = [rating for rating in similar_users if rating[1] > 0.95]

    # Get the stuff they rated, and add ratings for each item, weighted by user similarity
    candidates = defaultdict(float)
    for user in k_neighbors:
        inner_id, user_similarity = user[0], user[1]
        ratings = trainset.ur[inner_id]
        for rating in ratings:
            candidates[rating[0]] += (rating[1] / 5.0) * user_similarity

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
