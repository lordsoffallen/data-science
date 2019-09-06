from surprise.model_selection import train_test_split, LeaveOneOut
from surprise.dataset import DatasetAutoFolds
from surprise import KNNBaseline
from collections import defaultdict


class EvaluationData:
    def __init__(self, data, popularity_rankings):
        """ Init Data related variables to be used in evaluation.

        Parameters
        ----------
        data: DatasetAutoFolds
            Data which we are creating a model from. Should be variable derived from
            suprise Dataset class.
        popularity_rankings: defaultdict
            A dict contains the ranking of items
        """

        # Build a full training set for evaluating overall properties
        self.full_trainset = data.build_full_trainset()
        self.full_antiset = self.full_trainset.build_anti_testset()

        # Build a 75/25 train/test split for measuring accuracy
        self.trainset, self.testset = train_test_split(data, test_size=.25, random_state=1)
        
        # Build a "leave one out" train/test split for evaluating top-N recommenders
        # And build an anti-test-set for building predictions
        loocv = LeaveOneOut(n_splits=1, random_state=1)
        for train, test in loocv.split(data):
            self.loocv_train = train
            self.loocv_test = test
            
        self.loocv_anti_testset = self.loocv_train.build_anti_testset()
        self.rankings = popularity_rankings
        
        # Compute similarty matrix between items so we can measure diversity
        sim_options = {'name': 'cosine', 'user_based': False}
        self.sims_algo = KNNBaseline(sim_options=sim_options)
        self.sims_algo.fit(self.full_trainset)
            
    def get_full_trainset(self):
        return self.full_trainset
    
    def get_full_anti_testset(self):
        return self.full_antiset
    
    def get_anti_testset_for_user(self, test_subject):
        """

        Parameters
        ----------
        test_subject: int
            Test subject id

        Returns
        -------
        anti_testset: list
            Returns anti test set
        """

        trainset = self.full_trainset
        fill = trainset.global_mean
        anti_testset = []
        u = trainset.to_inner_uid(str(test_subject))
        user_items = set([j for j, _ in trainset.ur[u]])
        anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                                 i in trainset.all_items() if
                                 i not in user_items]
        return anti_testset

    def get_trainset(self):
        return self.trainset
    
    def get_testset(self):
        return self.testset
    
    def get_loocv_trainset(self):
        return self.loocv_train
    
    def get_loocv_testset(self):
        return self.loocv_test
    
    def get_loocv_anti_testset(self):
        return self.loocv_anti_testset
    
    def get_similarities(self):
        return self.sims_algo
    
    def get_popularity_rankings(self):
        return self.rankings
