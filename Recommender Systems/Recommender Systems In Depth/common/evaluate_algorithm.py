from common.metrics import RecommenderMetrics as metrics
from common.data import EvaluationData


class EvaluateAlgorithm:
    def __init__(self, algorithm, name):
        """ This class implements evaluate function which evaluates the
        different algorithms performance.

        Parameters
        ----------
        algorithm: Any
            The algorithm to fit the data. Should have implement fit() and
            test() functions
        name: str
            Name of the algorithm
        """

        self.algorithm = algorithm
        self.name = name

    def evaluate(self, data, eval_topn, n=10, verbose=True):
        """ Evalute the model performance on a given data.

        Parameters
        ----------
        data: EvaluationData
            A dataset constructed from EvaluationData class
        eval_topn: bool
            Indicates whether to apply Leave One Out Cross Validation.
        n: int
            If eval_topn is true, then this n parameter used to retrieve
            top N.
        verbose: bool
            Indicates the verbosity of the function

        Returns
        -------
        metrics: dict
            A dictionary contains all measured metrics.
        """

        if verbose: print("Evaluating accuracy...")

        eval_metrics = {}
        self.algorithm.fit(data.get_trainset())
        predictions = self.algorithm.test(data.get_testset())
        eval_metrics["RMSE"] = metrics.rmse(predictions)
        eval_metrics["MAE"] = metrics.mae(predictions)

        if eval_topn:
            # Evaluate top-10 with Leave One Out testing
            if verbose: print("Evaluating top-N with leave-one-out...")

            self.algorithm.fit(data.get_loocv_trainset())
            leftout_preds = self.algorithm.test(data.get_loocv_testset())

            # Build predictions for all ratings not in the training set
            all_preds = self.algorithm.test(data.get_loocv_anti_testset())

            # Compute top 10 recs for each user
            topn_pred = metrics.get_top_n(all_preds, n)

            if verbose: print("Computing hit-rate and rank metrics...")

            # See how often we recommended a movie the user actually rated
            eval_metrics["HR"] = metrics.hit_rate(topn_pred, leftout_preds)

            # See how often we recommended a movie the user actually liked
            eval_metrics["cHR"] = metrics.hit_rate(topn_pred, leftout_preds, rating_cutoff=0)

            # Compute ARHR
            eval_metrics["ARHR"] = metrics.avg_reciprocal_hit_rank(topn_pred, leftout_preds)

            # Evaluate properties of recommendations on full training set
            if verbose: print("Computing recommendations with full data set...")

            self.algorithm.fit(data.get_full_trainset())
            all_preds = self.algorithm.test(data.get_full_anti_testset())
            topn_pred = metrics.get_top_n(all_preds, n)

            if verbose: print("Analyzing coverage, diversity, and novelty...")

            # Print user coverage with a minimum predicted rating of 4.0:
            eval_metrics["Coverage"] = metrics.user_coverage(topn_pred, data.get_full_trainset().n_users, 4.0)

            # Measure diversity of recommendations:
            eval_metrics["Diversity"] = metrics.diversity(topn_pred, data.get_similarities())

            # Measure novelty (average popularity rank of recommendations):
            eval_metrics["Novelty"] = metrics.novelty(topn_pred, data.get_popularity_rankings())

        if verbose: print("Analysis complete.")

        return eval_metrics

    def get_name(self):
        return self.name

    def get_algorithm(self):
        return self.algorithm
