from surprise import AlgoBase, PredictionImpossible
from common.movielens import MovieLens
from collections import defaultdict
from heapq import nlargest
import math
import numpy as np



class ContentKNNAlgorithm(AlgoBase):
    def __init__(self, k=40, sim_options={}, verbose=False):
        """ Init a ContentKNNAlgorithm class to calculate item based
        recommendations.

        Parameters
        ----------
        k: int
            Number of nearest neighbors
        sim_options: dict
            A dict object contains similarity options for AlgoBase
        verbose: bool
            Verbosity of fit method.
        """
        AlgoBase.__init__(self, sim_options=sim_options)
        self.k = k
        self.verbose = verbose

    def fit(self, trainset):
        """ Fit an algorithm to a KNN model.

        Parameters
        ----------
        trainset:
            The data the used in training

        Returns
        -------
        model: ContentKNNAlgorithm
            Returns the class instance
        """

        AlgoBase.fit(self, trainset)

        # Compute item similarity matrix based on content attributes
        # Load up genre vectors for every movie
        ml = MovieLens()
        genres = ml.get_genres()
        years = ml.get_years()
        
        print("Computing content-based similarity matrix...")
            
        # Compute genre distance for every movie combination as a 2x2 matrix
        self.similarities = np.zeros((self.trainset.n_items, self.trainset.n_items))
        
        for rating1 in range(self.trainset.n_items):
            if rating1 % 100 == 0 and self.verbose:
                print(rating1, " of ", self.trainset.n_items)

            for rating2 in range(rating1+1, self.trainset.n_items):
                movie_id1 = int(self.trainset.to_raw_iid(rating1))
                movie_id2 = int(self.trainset.to_raw_iid(rating2))
                genre_similarity = self.genre_similarity(movie_id1, movie_id2, genres)
                year_similarity = self.year_similarity(movie_id1, movie_id2, years)
                self.similarities[rating1, rating2] = genre_similarity * year_similarity
                self.similarities[rating2, rating1] = self.similarities[rating1, rating2]
                
        print("...done.")
                
        return self
    
    @staticmethod
    def genre_similarity(movie_id1, movie_id2, genres):
        """ This function computes the genre similarities between
        two different movies.

        Parameters
        ----------
        movie_id1: int
            First movie ID
        movie_id2: int
            Other movie ID
        genres: defaultdict
            A dict of lists contains genres for each movie id

        Returns
        -------
        sim_genre: float
            Returns genre similarity scores
        """

        genres1, genres2 = genres[movie_id1], genres[movie_id2]
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(genres1)):
            x, y = genres1[i], genres2[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y
        
        return sumxy/math.sqrt(sumxx*sumyy)
    
    @staticmethod
    def year_similarity(movie_id1, movie_id2, years):
        """ This function computes the year similarities between
        two different movies.

        Parameters
        ----------
        movie_id1: int
            First movie ID
        movie_id2: int
            Other movie ID
        years: defaultdict
            A dict contains years for each movie id

        Returns
        -------
        sim_year: float
            Returns year similarity scores
        """

        diff = abs(years[movie_id1] - years[movie_id2])
        sim = math.exp(-diff / 10.0)
        return sim
    
    @staticmethod
    def mise_en_scene_similarity(movie_id1, movie_id2, mes):
        """ This function calculates the mise en scene similarity based on
        their mise en scene data.

        Parameters
        ----------
        movie_id1: int
            First movie ID
        movie_id2: int
            Other movie ID
        mes: defaultdict
            A dict of list contains mise en scene information

        Returns
        -------
        mes: float
            Returns the similarty between two items.
        """

        mes1 = mes[movie_id1]
        mes2 = mes[movie_id2]
        if mes1 and mes2:
            shot_length_diff = math.fabs(mes1[0] - mes2[0])
            color_variance_diff = math.fabs(mes1[1] - mes2[1])
            motion_diff = math.fabs(mes1[3] - mes2[3])
            lighting_diff = math.fabs(mes1[5] - mes2[5])
            num_shots_diff = math.fabs(mes1[6] - mes2[6])
            return shot_length_diff * color_variance_diff * motion_diff * lighting_diff * num_shots_diff
        else:
            return 0

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
        
        # Build up similarity scores between this item and everything the user rated
        neighbors = []
        for rating in self.trainset.ur[u]:
            genre_similarity = self.similarities[i, rating[0]]
            neighbors.append((genre_similarity, rating[1]))
        
        # Extract the top-K most-similar ratings
        k_neighbors = nlargest(self.k, neighbors, key=lambda t: t[0])
        
        # Compute average sim score of K neighbors weighted by user ratings
        sim_total = weighted_sum = 0
        for sim_score, rating in k_neighbors:
            if sim_score > 0:
                sim_total += sim_score
                weighted_sum += sim_score * rating
            
        if sim_total == 0:
            raise PredictionImpossible('No neighbors')

        return weighted_sum / sim_total
