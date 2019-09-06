from surprise import accuracy
from collections import defaultdict
from itertools import combinations


class RecommenderMetrics:

    @staticmethod
    def mae(predictions):
        return accuracy.mae(predictions, verbose=False)

    @staticmethod
    def rmse(predictions):
        return accuracy.rmse(predictions, verbose=False)

    @staticmethod
    def get_top_n(predictions, n=10, min_rating=4.0):
        """ Get the top n predicted movies based on their sorted ratings
        
        Parameters
        ----------
        predictions: list
            Predictions from a model that consists user_id, movie_id, 
            actual_rating, estimated_rating within.
        n: int
            Number of movies to retrieve
        min_rating: float
            Minimum rating to consider when filtering the predictions

        Returns
        -------
        topn: defaultdict
            A top n predictions to recommend in user_id as key, (movie id, ratings)
            as value format
        """
        
        top_n = defaultdict(list)

        for user_id, movie_id, actual_rating, estimated_rating, _ in predictions:
            if estimated_rating >= min_rating:
                top_n[int(user_id)].append((int(movie_id), estimated_rating))

        for user_id, ratings in top_n.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[int(user_id)] = ratings[:n]

        return top_n

    @staticmethod
    def hit_rate(topn_pred, leftout_pred, rating_cutoff=None):
        """ Calcuate the hit ratio of recommended predictions.It is 
        calculated by dividing the correctly predicted movies over total
        predictions.

        Parameters
        ----------
        topn_pred: defaultdict
            Top N predictions in user_id as key, (movie id, ratings)
            as value format
        leftout_pred: list
            Test predictions to calculate the hit rate in user id, movie id format.
        rating_cutoff: None, float
            A float to compare ratings. Ratings below this one will not be accumulated
            in metric computation. 
            
        Returns
        -------
        precision: float
            Returns the overall hit rate precision. If rating_cutoff is given, returns
            the cumulative hit rate.
        """
        
        hits, total = 0, 0
        if rating_cutoff is None:  # Not cumulative
            # For each left-out rating
            for user_id, leftout_movie_id, actual_rating, estimated_rating, _ in leftout_pred:
                # Is it in the predicted top 10 for this user?
                for movie_id, predicted_rating in topn_pred[int(user_id)]:
                    if int(leftout_movie_id) == int(movie_id):
                        hits += 1
                        break
                total += 1
        else:
            # For each left-out rating
            for user_id, leftout_movie_id, actual_rating, estimated_rating, _ in leftout_pred:
                # Only look at ability to recommend things the users actually liked...
                if actual_rating >= rating_cutoff:
                    for movie_id, predicted_rating in topn_pred[int(user_id)]:
                        if int(leftout_movie_id) == movie_id:
                            hits += 1
                            break
                    total += 1

        # Compute overall precision
        return hits/total

    @staticmethod
    def rating_hit_rate(topn_pred, leftout_pred):
        """ Prints rating and its hit rate 
        
        Parameters
        ----------
        topn_pred: defaultdict
            Top N predictions in user_id as key, (movie id, ratings)
            as value format
        leftout_pred: list
            Test predictions to calculate the hit rate in user id, movie id format.
        """
        
        hits, total = defaultdict(float), defaultdict(float)

        # For each left-out rating
        for user_id, leftout_movie_id, actual_rating, estimated_rating, _ in leftout_pred:
            # Is it in the predicted top N for this user?
            for movie_id, predicted_rating in topn_pred[int(user_id)]:
                if int(leftout_movie_id) == movie_id:
                    hits[actual_rating] += 1
                    break    
            total[actual_rating] += 1

        # Compute overall precision
        for rating in sorted(hits.keys()):
            print(rating, hits[rating] / total[rating])

    @staticmethod
    def avg_reciprocal_hit_rank(topn_pred, leftout_pred):
        """ Calculates the average reciprocal hit ranking.
        
        Parameters
        ----------
        topn_pred: defaultdict
            Top N predictions in user_id as key, (movie id, ratings)
            as value format
        leftout_pred: list
            Test predictions to calculate the hit rate in user id, movie id format.

        Returns
        -------
        avg_rank: float
            Returns the avg hit ranking
        """
        
        summation, total = 0, 0
        
        # For each left-out rating
        for user_id, leftout_movie_id, actual_rating, estimated_rating, _ in leftout_pred:
            # Is it in the predicted top N for this user?
            rank = 0
            for movie_id, predicted_rating in topn_pred[int(user_id)]:
                rank = rank + 1
                if int(leftout_movie_id) == movie_id:
                    summation += 1.0 / rank
                    break
            total += 1

        return summation / total

    @staticmethod
    def user_coverage(topn_pred, num_user, rating_threshold=0):
        """ Calculate the user coverage by diving the hits over number
        of users. We are trying to measure what percentage of users have
        at least one "good" recommendation

        Parameters
        ----------
        topn_pred: defaultdict
            Top N predictions in user_id as key, (movie id, ratings)
            as value format
        num_user: int
            Number of users to consider
        rating_threshold: float
            A threshold value to cut off unwanted ratings

        Returns
        -------
        coverage: float
            Returns user coverage
        """

        hits = 0
        for user_id in topn_pred.keys():
            for movie_id, predicted_rating in topn_pred[user_id]:
                if predicted_rating >= rating_threshold:
                    hits += 1
                    break

        return hits / num_user

    @staticmethod
    def diversity(topn_pred, model):
        """ Compute the average similarity between recommendation pairs.

        Parameters
        ----------
        topn_pred: defaultdict
            Top N predictions in user_id as key, (movie id, ratings)
            as value format
        model:
            A class used to train the data.
        Returns
        -------
        diversity: float
            Returns diversity metrics
        """

        n, total = 0, 0
        sim_matrix = model.compute_similarities()

        for user_id in topn_pred.keys():
            pairs = combinations(topn_pred[user_id], 2)
            for pair in pairs:
                movie1, movie2 = pair[0][0], pair[1][0]
                inner_id1 = model.trainset.to_inner_iid(str(movie1))
                inner_id2 = model.trainset.to_inner_iid(str(movie2))
                similarity = sim_matrix[inner_id1][inner_id2]
                total += similarity
                n += 1

        S = total / n
        return 1 - S

    @staticmethod
    def novelty(topn_pred, rankings):
        """ Calculate the mean popularity of recommended items

        Parameters
        ----------
        topn_pred: defaultdict
            Top N predictions in user_id as key, (movie id, ratings)
            as value format
        rankings: defaultdict
            Movies sorted by their ranks

        Returns
        -------
        novelty: float
            Returns novelty metric
        """
        n, total = 0, 0
        for user_id in topn_pred.keys():
            for rating in topn_pred[user_id]:
                movie_id = rating[0]
                rank = rankings[movie_id]
                total += rank
                n += 1
        return total / n
