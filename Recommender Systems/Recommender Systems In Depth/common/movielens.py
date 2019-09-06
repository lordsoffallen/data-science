from collections import defaultdict
from surprise import Dataset
from surprise.dataset import DatasetAutoFolds
from surprise import Reader
import os, csv, sys, re


class MovieLens:
    def __init__(self):
        self.id2name = {}
        self.name2id = {}
        self.ratings = '../ml-latest-small/ratings.csv'
        self.movies = '../ml-latest-small/movies.csv'
    
    def load(self):
        """ Loads the small movielens ratings dataset and constructs the dict mappings
        
        Returns
        -------
        ds: DatasetAutoFolds
            Constructed dataset from ratings
        """
        
        # Look for files relative to the directory we are running from
        os.chdir(os.path.dirname(sys.argv[0]))
        reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
        ratings_ds = Dataset.load_from_file(self.ratings, reader=reader)

        with open(self.movies, newline='', encoding='ISO-8859-1') as f:
            movie_reader = csv.reader(f)
            next(movie_reader)  # Skip header line
            for row in movie_reader:
                id, name = int(row[0]), row[1]
                self.id2name[id] = name
                self.name2id[name] = id

        return ratings_ds

    def get_user_ratings(self, user):
        """ Retrieve the user ratings for a given user id
        
        Parameters
        ----------
        user: int
            User id
            
        Returns
        -------
        user_ratings: list
            Returns found user ratings if any
        """
        
        user_ratings = []
        hit_user = False

        with open(self.ratings, newline='') as csvfile:
            rating_reader = csv.reader(csvfile)
            next(rating_reader)
            for row in rating_reader:
                user_id = int(row[0])
                if user == user_id:
                    movie_id, rating = int(row[1]), float(row[2])
                    user_ratings.append((movie_id, rating))
                    hit_user = True
                if hit_user and user != user_id:
                    break

        return user_ratings

    def get_popularity_ranks(self):
        """ Get popularity ranks for movies. 
        
        Returns
        -------
        rankings: defaultdict
            Returns movies sorted by their ranks
        """
        
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        
        # Count the number of ratings for movies
        with open(self.ratings, newline='') as f:
            rating_reader = csv.reader(f)
            next(rating_reader)
            for row in rating_reader:
                movie_id = int(row[1])
                ratings[movie_id] += 1
                
        rank = 1
        for movie_id, rating_count in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
            rankings[movie_id] = rank
            rank += 1
            
        return rankings
    
    def get_genres(self):
        """ Get the unique genres from the data

        Returns
        -------
        genres: defaultdict
            Returns the genres from the data
        """

        genres = defaultdict(list)
        genre_ids = {}
        max_genre_id = 0

        with open(self.movies, newline='', encoding='ISO-8859-1') as f:
            movie_reader = csv.reader(f)
            next(movie_reader)
            for row in movie_reader:
                movie_id, genre_list = int(row[0]), row[2].split('|')
                genre_id_list = []
                for genre in genre_list:
                    if genre in genre_ids:
                        genre_id = genre_ids[genre]
                    else:
                        genre_id = max_genre_id
                        genre_ids[genre] = genre_id
                        max_genre_id += 1
                    genre_id_list.append(genre_id)
                genres[movie_id] = genre_id_list

        # Convert integer-encoded genre lists to bitfields that we can treat as vectors
        for (movie_id, genre_id_list) in genres.items():
            bitfield = [0] * max_genre_id
            for genre_id in genre_id_list:
                bitfield[genre_id] = 1
            genres[movie_id] = bitfield            
        
        return genres
    
    def get_years(self):
        """ Get the years from timestamp column

        Returns
        -------
        years: defaultdict
            A movie id year pair
        """

        p = re.compile(r"(?:\((\d{4})\))?\s*$")
        years = defaultdict(int)
        with open(self.movies, newline='', encoding='ISO-8859-1') as f:
            movie_reader = csv.reader(f)
            next(movie_reader)
            for row in movie_reader:
                movie_id, title = int(row[0]), row[1]
                m = p.search(title)
                year = m.group(1)
                if year:
                    years[movie_id] = int(year)
        return years

    def get_movie_name(self, movie_id):

        if movie_id in self.id2name:
            return self.id2name[movie_id]
        else:
            return ""

    def get_movie_id(self, movie_name):
        if movie_name in self.name2id:
            return self.name2id[movie_name]
        else:
            return 0

    @staticmethod
    def get_mise_en_scene():
        """ This function retrieves the mise en scene data. This is
        a bleeding edge method and doesnt affect the accuracy but rather
        improves diversity

        Returns
        -------
        mes: defaultdict
            A dict of lists contains mise en scene info
        """

        mes = defaultdict(list)

        with open("LLVisualFeatures13K_Log.csv", newline='') as f:
            mes_reader = csv.reader(f)
            next(mes_reader)
            for row in mes_reader:
                movie_id = int(row[0])
                avg_shot_length = float(row[1])
                mean_color_variance = float(row[2])
                stddev_color_variance = float(row[3])
                mean_motion = float(row[4])
                stddev_motion = float(row[5])
                mean_lighting_key = float(row[6])
                num_shots = float(row[7])
                mes[movie_id] = [avg_shot_length, mean_color_variance, stddev_color_variance,
                                 mean_motion, stddev_motion, mean_lighting_key, num_shots]

        return mes
