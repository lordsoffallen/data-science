from common.data import EvaluationData
from common.evaluate_algorithm import EvaluateAlgorithm
from common.movielens import MovieLens
from collections import defaultdict
from surprise.dataset import DatasetAutoFolds


class Evaluator:
    def __init__(self, dataset, rankings):
        """ Init a evaluate object to evaluate different algorithms.
        First use add_algorithm function to add some models and then
        call evaluate method off of it to print the metrics.

        Parameters
        ----------
        dataset: DatasetAutoFolds
           Data which we are creating a model from. Should be variable derived from
           suprise Dataset class.
        rankings: defaultdict
           A dict contains the ranking of items
        """

        self.dataset = EvaluationData(dataset, rankings)
        self.algorithms = []
        
    def add_algorithm(self, algorithm, name):
        """ Add an algorithm type to evaluate

        Parameters
        ----------
        algorithm: Any
            The algorithm to fit the data. Should have implement fit() and
            test() functions
        name: str
            Name of the algorithm
        """
        self.algorithms.append(EvaluateAlgorithm(algorithm, name))
        
    def evaluate(self, eval_topn):
        """ This function evalutes the model performance on different metrics
        and print them.

        Parameters
        ----------
        eval_topn: bool
            Indicates whether to apply Leave One Out Cross Validation.
        """

        results = {}
        for algorithm in self.algorithms:
            print("Evaluating ", algorithm.get_name(), "...")
            results[algorithm.get_name()] = algorithm.evaluate(self.dataset, eval_topn)

        # Print results
        print("\n")
        
        if eval_topn:
            print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                    "Algorithm", "RMSE", "MAE", "HR", "cHR", "ARHR", "Coverage", "Diversity", "Novelty"))
            for name, metrics in results.items():
                print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                        name, metrics["RMSE"], metrics["MAE"], metrics["HR"], metrics["cHR"], metrics["ARHR"],
                        metrics["Coverage"], metrics["Diversity"], metrics["Novelty"]))
        else:
            print("{:<10} {:<10} {:<10}".format("Algorithm", "RMSE", "MAE"))
            for name, metrics in results.items():
                print("{:<10} {:<10.4f} {:<10.4f}".format(name, metrics["RMSE"], metrics["MAE"]))
                
        print("\nLegend:\n")
        print("RMSE:      Root Mean Squared Error. Lower values mean better accuracy.")
        print("MAE:       Mean Absolute Error. Lower values mean better accuracy.")
        if eval_topn:
            print("HR:        Hit Rate; how often we are able to recommend a left-out rating. Higher is better.")
            print("cHR:       Cumulative Hit Rate; hit rate, confined to ratings above a certain threshold. "
                  "Higher is better.")
            print("ARHR:      Average Reciprocal Hit Rank - Hit rate that takes the ranking into account. "
                  "Higher is better.")
            print("Coverage:  Ratio of users for whom recommendations above a certain threshold exist. "
                  "Higher is better.")
            print("Diversity: 1-S, where S is the average similarity score between every possible pair "
                  "of recommendations for a given user. Higher means more diverse.")
            print("Novelty:   Average popularity rank of recommended items. Higher means more novel.")
        
    def sample_topn_recs(self, ml, subject=85, k=10):
        """

        Parameters
        ----------
        ml: MovieLens
            An instance of MovieLens class to retrieve movie names
        subject: int
            Test subject id
        k: int
            Number of recommendation to sample
        """

        for algo in self.algorithms:
            print("\nUsing recommender ", algo.get_name())
            print("\nBuilding recommendation model...")
            trainset = self.dataset.get_full_trainset()
            algo.get_algorithm().fit(trainset)
            
            print("Computing recommendations...")
            testset = self.dataset.get_anti_testset_for_user(subject)
            predictions = algo.get_algorithm().test(testset)

            recommendations = []
            
            print ("\nWe recommend:")
            for user_id, movie_id, actual_rating, estimated_rating, _ in predictions:
                recommendations.append((int(movie_id), estimated_rating))

            recommendations.sort(key=lambda x: x[2])
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            for ratings in recommendations[:k]:
                print(ml.get_movie_name(ratings[0]), ratings[1])
