from common.movielens import MovieLens
from surprise import SVD, KNNBaseline
from surprise.model_selection import train_test_split, LeaveOneOut
from common.metrics import RecommenderMetrics as metrics


if __name__ == '__main__':
    ml = MovieLens()

    print("Loading movie ratings...")
    data = ml.load()
    
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.get_popularity_ranks()
    
    print("\nComputing item similarities so we can measure diversity later...")
    full_trainset = data.build_full_trainset()
    options = {'name': 'pearson_baseline', 'user_based': False}
    knn_model = KNNBaseline(sim_options=options)
    knn_model.fit(full_trainset)
    
    print("\nBuilding recommendation model...")
    train, test = train_test_split(data, test_size=.25, random_state=1)
    
    svd_model = SVD(random_state=10)
    svd_model.fit(train)
    
    print("\nComputing recommendations...")
    predictions = svd_model.test(test)
    
    print("\nEvaluating accuracy of model...")
    print("RMSE: ", metrics.rmse(predictions))
    print("MAE: ", metrics.mae(predictions))
    
    print("\nEvaluating top-10 recommendations...")
    
    # Set aside one rating per user for testing
    LOOCV = LeaveOneOut(n_splits=1, random_state=1)
    
    for train, test in LOOCV.split(data):
        print("Computing recommendations with leave-one-out...")
    
        # Train model without left-out ratings
        svd_model.fit(train)
    
        # Predicts ratings for left-out ratings only
        print("Predict ratings for left-out set...")
        leftout_pred = svd_model.test(test)
    
        # Build predictions for all ratings not in the training set
        print("Predict all missing ratings...")
        big_testset = train.build_anti_testset()
        all_preds = svd_model.test(big_testset)
    
        # Compute top 10 recs for each user
        print("Compute top 10 recs per user...")
        topn_pred = metrics.get_top_n(all_preds, n=10)
    
        # See how often we recommended a movie the user actually rated
        print("\nHit Rate: ", metrics.hit_rate(topn_pred, leftout_pred))
    
        # Break down hit rate by rating value
        print("\nrHR (Hit Rate by Rating value): ")
        metrics.rating_hit_rate(topn_pred, leftout_pred)
    
        # See how often we recommended a movie the user actually liked
        print("\ncHR (Cumulative Hit Rate, rating >= 4): ", metrics.hit_rate(topn_pred, leftout_pred, 4.0))
    
        # Compute ARHR
        print("\nARHR (Average Reciprocal Hit Rank): ", metrics.avg_reciprocal_hit_rank(topn_pred, leftout_pred))
    
    print("\nComputing complete recommendations, no hold outs...")
    svd_model.fit(full_trainset)
    big_testset = full_trainset.build_anti_testset()
    all_preds = svd_model.test(big_testset)
    topn_pred = metrics.get_top_n(all_preds, n=10)
    
    # Print user coverage with a minimum predicted rating of 4.0:
    print("\nUser coverage: ", metrics.user_coverage(topn_pred, num_user=full_trainset.n_users, rating_threshold=4.0))
    
    # Measure diversity of recommendations:
    print("\nDiversity: ", metrics.diversity(topn_pred, knn_model))
    
    # Measure novelty (average popularity rank of recommendations):
    print("\nNovelty (average popularity rank): ", metrics.novelty(topn_pred, rankings))
