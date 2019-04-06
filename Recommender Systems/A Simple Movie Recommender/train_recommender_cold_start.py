import numpy as np
import pandas as pd
import pickle
import util.matrix_factorization as factor

# Load user ratings
df = pd.read_csv('movie_ratings_data_set.csv')

# Convert the running list of user ratings into a matrix
ratings_df = df.pivot_table(index='user_id', columns='movie_id', aggfunc=np.max)

# Normalize the ratings (center them around their mean) to make it
# work for first time users
normalized_ratings, means = factor.normalize_ratings(ratings_df.values)

# Apply matrix factorization to find the latent features
U, M = factor.low_rank_matrix_factorization(normalized_ratings,
                                            num_features=11,
                                            regularization_amount=1.1)

# Find all predicted ratings by multiplying U and M
predicted_ratings = np.matmul(U, M)

# Add back in the mean ratings for each product to de-normalize the predicted results
predicted_ratings = predicted_ratings + means

# Save features and predicted ratings to files for later use
pickle.dump(U, open("user_features.dat", "wb"))
pickle.dump(M, open("product_features.dat", "wb"))
pickle.dump(predicted_ratings, open("predicted_ratings.dat", "wb" ))
pickle.dump(means, open("means.dat", "wb" ))
