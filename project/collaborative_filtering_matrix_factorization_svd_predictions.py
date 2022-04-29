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
import copy

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

# Load data
start = time.time()
reader = Reader()
data = Dataset.load_from_df(new_transactions_train_min_max_scaled_df[['customer_id', 'article_id', 'rating']], reader)
print("Time taken to load data: %s seconds" % (time.time() - start))

# Build full trainset
start = time.time()
trainset = data.build_full_trainset()
print("Time taken to build full trainset: %s seconds" % (time.time() - start))

# Matrix Factorization algorithm - SVD - Fit trainset
start = time.time()
algo = SVD()
algo.fit(trainset)
print("Time taken to fit trainset for SVD: %s seconds" % (time.time() - start))

# Make a deep-copy of the trainset, as a working copy
users_ratings_dict = defaultdict(list)
users_ratings_dict = copy.deepcopy(trainset.ur)

IS_TEST = str(input())

# Compute predictions and post-process
list_customer_id = []
list_prediction = []
NUM_RECOMMENDATIONS_PER_CUSTOMER = 12

top_selling_articles_by_number_array = transactions_train_df.groupby('article_id').count()['t_dat'].sort_values(ascending=False).head(NUM_RECOMMENDATIONS_PER_CUSTOMER)

if IS_TEST.lower() == 'true':
    iterations = 0
    MAX_ITERATIONS_FOR_TEST = 10
    
for uid, user_ratings in users_ratings_dict.items():
    if IS_TEST.lower() == 'true':
        iterations += 1
        if iterations >= (MAX_ITERATIONS_FOR_TEST+1):
            break
    customer_id = trainset.to_raw_uid(uid)
    user_ratings.sort(key=lambda x: x[1], reverse=True)
    users_ratings_dict[uid] = user_ratings[:]
    list_customer_id.append(customer_id)
    recommended_articles_string = ''
    for i in users_ratings_dict[uid]:
        (curr_article_id, curr_rating) = i
        recommended_articles_string += str(trainset.to_raw_iid(curr_article_id))
        recommended_articles_string += str(' ')
    for i, curr_article_id in enumerate(top_selling_articles_by_number_array.index):
        recommended_articles_string += str(curr_article_id)
        recommended_articles_string += str(' ')  
    recommended_articles_list = recommended_articles_string.split()
    purchased_articles_array = transactions_train_df.loc[transactions_train_df['customer_id'] == customer_id, 'article_id'].values
    for i in purchased_articles_array:
        if i in recommended_articles_list:
            recommended_articles_list.remove(i)  
    unique_recommended_articles = set(recommended_articles_list)
    list_prediction.append((' '.join(unique_recommended_articles))[:119]) # list is truncated to 12 articles each of length 9, space-separated, hence, 12*9+11

for row in customers_df.itertuples():
    if IS_TEST.lower() == 'true':
        iterations += 1
        if iterations >= (MAX_ITERATIONS_FOR_TEST+1):
            break
    customer_id = getattr(row, 'customer_id')
    if customer_id not in transactions_train_df.values:
        list_customer_id.append(customer_id)
        recommended_articles_string = ''
        for i, curr_article_id in enumerate(top_selling_articles_by_number_array.index):
            recommended_articles_string += str(curr_article_id)
            recommended_articles_string += str(' ')  
        list_prediction.append(recommended_articles_string[:119]) # list is truncated to 12 articles each of length 9, space-separated, hence, 12*9+11    

submission_content_based_detail_desc_df = pd.DataFrame(list(zip(list_customer_id, list_prediction)), columns =['customer_id', 'prediction'])
submission_content_based_detail_desc_df.to_csv('submission_collaborative_filtering_collaborative_filtering_matrix_factorization_svd_predictions.csv', encoding='utf-8', index=False)
submission_content_based_detail_desc_df
