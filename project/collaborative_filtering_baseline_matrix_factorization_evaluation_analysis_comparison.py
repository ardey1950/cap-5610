import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surprise import Reader, Dataset
from surprise import BaselineOnly
from surprise import NormalPredictor
from surprise import SVD
from surprise import SVDpp
from surprise import KNNBasic
from surprise.model_selection import cross_validate
from surprise import accuracy
from surprise.model_selection import train_test_split
import time
import math
from collections import defaultdict

import warnings; warnings.simplefilter('ignore')

# Record start time before reading dataset
start = time.time()
articles_df = pd.read_csv('../h-and-m-personalized-fashion-recommendations/articles.csv')
customers_df = pd.read_csv('../h-and-m-personalized-fashion-recommendations/customers.csv')
transactions_train_df = pd.read_csv('../h-and-m-personalized-fashion-recommendations/transactions_train.csv')
# Print time taken to read dataset
print("Time taken to read dataset: %s seconds" % (time.time() - start))

count_series = transactions_train_df.groupby(['customer_id', 'article_id']).size()

# For a customer_id, rating of an article_id is the number of purchas(s) of the article by the customer
new_transactions_train_df = count_series.to_frame(name='rating').reset_index()
new_transactions_train_df

# Applying min-max normalization to the 'rating' column to scale the rating in the [1, 5] range
new_transactions_train_min_max_scaled_df = new_transactions_train_df.copy()
new_transactions_train_min_max_scaled_df['rating'] = 1 + (((new_transactions_train_min_max_scaled_df['rating'] - new_transactions_train_min_max_scaled_df['rating'].min()) * (5 - 1)) / (new_transactions_train_min_max_scaled_df['rating'].max() - new_transactions_train_min_max_scaled_df['rating'].min()))
new_transactions_train_min_max_scaled_df

start = time.time()
reader = Reader()
data = Dataset.load_from_df(new_transactions_train_min_max_scaled_df[['customer_id', 'article_id', 'rating']], reader)
print("Time taken to load data: %s seconds" % (time.time() - start))

# Baseline algorithm - Normal Predictor
start = time.time()
algo = NormalPredictor()
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("Time taken to cross-validate NormalPredictor: %s seconds" % (time.time() - start))

# Baseline algorithm - Baseline Only
start = time.time()
algo = BaselineOnly()
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("Time taken to cross-validate BaselineOnly: %s seconds" % (time.time() - start))

# Matrix Factorization-based algorithm - Probabilistic Matrix Factorization (PMF)
start = time.time()
algo = SVD(biased=False)
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("Time taken to cross-validate PMF: %s seconds" % (time.time() - start))

# Matrix Factorization-based algorithm - Single Value Decomposition (SVD)
start = time.time()
algo = SVD()
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("Time taken to cross-validate SVD: %s seconds" % (time.time() - start))
