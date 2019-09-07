from surprise import AlgoBase
from surprise import PredictionImpossible
import numpy as np
from .autorec import AutoRec


class AutoRecAlgorithm(AlgoBase):

    def __init__(self, epochs=100, hidden_dim=100, learning_rate=0.01, batch_size=100, sim_options={}):
        AlgoBase.__init__(self, sim_options=sim_options)
        self.epochs = epochs
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def fit(self, trainset):
        """ Fit an algorithm to a RBM model.

        Parameters
        ----------
        trainset:
            The data the used in training

        Returns
        -------
        model: AutoRecAlgorithm
            Returns the class instance
        """

        AlgoBase.fit(self, trainset)

        num_users = trainset.n_users
        num_items = trainset.n_items

        train_matrix = np.zeros([num_users, num_items, 10], dtype=np.float32)

        for uid, iid, rating in trainset.all_ratings():
            train_matrix[int(uid), int(iid)] = rating / 5.0
        
        # Create an AutoRec with (num items * rating values) visible nodes
        autorec = AutoRec(train_matrix.shape[1], hidden_dimensions=self.hidden_dim,
                          learning_rate=self.learning_rate, batch_size=self.batch_size, epochs=self.epochs)
        autorec.train(train_matrix)

        self.predicted_ratings = np.zeros([num_users, num_items], dtype=np.float32)
        
        for uiid in range(trainset.n_users):
            if uiid % 50 == 0: print("Processing user ", uiid)

            recs = autorec.get_recommendations([train_matrix[uiid]])
            for item_id, rec in enumerate(recs):
                self.predicted_ratings[uiid, item_id] = rec * 5.0
        
        return self

    def estimate(self, u, i):
        """ Estimate a rating when given an user and an item.

        Parameters
        ----------
        u: int
            User id
        i: int
            Item id

        Returns
        -------
        rating: float
            Return a predicted rating for user, item pair.
        """

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        rating = self.predicted_ratings[u, i]

        if rating < 0.001:
            raise PredictionImpossible('No valid prediction exists.')

        return rating