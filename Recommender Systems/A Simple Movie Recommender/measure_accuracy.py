import numpy as np
import pandas as pd
import util.matrix_factorization as factor


# Load user ratings
training_df = pd.read_csv('movie_ratings_data_set_training.csv')
testing_df = pd.read_csv('movie_ratings_data_set_testing.csv')

# Convert the running list of user ratings into a matrix
ratings_training_df = training_df.pivot_table(index='user_id', columns='movie_id', aggfunc=np.max)
ratings_testing_df = testing_df.pivot_table(index='user_id', columns='movie_id', aggfunc=np.max)

# Apply matrix factorization to find the latent features
U, M = factor.low_rank_matrix_factorization(ratings_training_df.values,
                                            num_features=11,
                                            regularization_amount=1.1)

# Find all predicted ratings by multiplying U and M
predicted_ratings = np.matmul(U, M)

# Measure RMSE
rmse_training = factor.RMSE(ratings_training_df.values, predicted_ratings)
rmse_testing = factor.RMSE(ratings_testing_df.values, predicted_ratings)

print("Training RMSE: {}".format(rmse_training))
print("Testing RMSE: {}".format(rmse_testing))
