from surprise import AlgoBase, PredictionImpossible
from .rbm import RBM
import numpy as np


class RBMAlgorithm(AlgoBase):

    def __init__(self, epochs=20, hidden_dim=100, learning_rate=0.001, batch_size=100, sim_options={}):
        AlgoBase.__init__(self, sim_options=sim_options)
        self.epochs = epochs
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
    @staticmethod
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def fit(self, trainset):
        """ Fit an algorithm to a RBM model.

        Parameters
        ----------
        trainset:
            The data the used in training

        Returns
        -------
        model: RBMAlgorithm
            Returns the class instance
        """

        AlgoBase.fit(self, trainset)

        num_users = trainset.n_users
        num_items = trainset.n_items
        
        train_matrix = np.zeros([num_users, num_items, 10], dtype=np.float32)
        
        for uid, iid, rating in trainset.all_ratings():
            adjusted_rating = int(float(rating)*2.0) - 1
            train_matrix[int(uid), int(iid), adjusted_rating] = 1
        
        # Flatten to a 2D array, with nodes for each possible rating type on each possible item, for every user.
        train_matrix = np.reshape(train_matrix, [train_matrix.shape[0], -1])
        
        # Create an RBM with (num items * rating values) visible nodes
        rbm = RBM(train_matrix.shape[1], hidden_dimensions=self.hidden_dim,
                  learning_rate=self.learning_rate, batch_size=self.batch_size, epochs=self.epochs)
        rbm.train(train_matrix)

        self.predicted_ratings = np.zeros([num_users, num_items], dtype=np.float32)

        for uiid in range(trainset.n_users):
            if uiid % 50 == 0: print("Processing user ", uiid)

            recs = rbm.get_recommendations([train_matrix[uiid]])
            recs = np.reshape(recs, [num_items, 10])
            
            for item_id, rec in enumerate(recs):
                # The obvious thing would be to just take the rating with the highest score:                
                # rating = rec.argmax()
                # ... but this just leads to a huge multi-way tie for 5-star predictions.
                # The paper suggests performing normalization over K values to get probabilities
                # and take the expectation as our prediction.
                normalized = self.softmax(rec)
                rating = np.average(np.arange(10), weights=normalized)
                self.predicted_ratings[uiid, item_id] = (rating + 1) * 0.5
        
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
