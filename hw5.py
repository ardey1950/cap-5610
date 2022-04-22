import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surprise import Reader, Dataset
from surprise import SVD
from surprise import KNNBasic
from surprise.model_selection import cross_validate

import warnings; warnings.simplefilter('ignore')

reader = Reader()

# 'ratings_small.csv' is used
ratings = pd.read_csv('./hw5/ratings_small.csv')
print(ratings.head())

data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Probabilistic Matrix Factorization (PMF)
algo = SVD(biased=False)
dict_pmf = cross_validate(algo, data, measures=['rmse', 'mae'], cv=5, verbose=True)
print(dict_pmf)

# User based Collaborative Filtering using MSD similarity matrix
algo = KNNBasic()
dict_user_msd = cross_validate(algo, data, measures=['rmse', 'mae'], cv=5, verbose=True)
print(dict_user_msd)

# Item based Collaborative Filtering using MSD similarity matrix
sim_options = {'user_based': False}
algo = KNNBasic(sim_options=sim_options)
dict_item_msd = cross_validate(algo, data, measures=['rmse', 'mae'], cv=5, verbose=True)
print(dict_item_msd)

# User based Collaborative Filtering using Cosine similarity matrix
sim_options = {'name': 'cosine'}
algo = KNNBasic(sim_options=sim_options)
dict_user_cosine = cross_validate(algo, data, measures=['rmse', 'mae'], cv=5, verbose=True)
print(dict_user_cosine)

# User based Collaborative Filtering using Pearson similarity matrix
sim_options = {'name': 'pearson_baseline'}
algo = KNNBasic(sim_options=sim_options)
dict_user_pearson = cross_validate(algo, data, measures=['rmse', 'mae'], cv=5, verbose=True)
print(dict_user_pearson)

# Item based Collaborative Filtering using Cosine similarity matrix
sim_options = {'name': 'cosine', 'user_based': False}
algo = KNNBasic(sim_options=sim_options)
dict_item_cosine = cross_validate(algo, data, measures=['rmse', 'mae'], cv=5, verbose=True)
print(dict_item_cosine)

# Item based Collaborative Filtering using Pearson similarity matrix
sim_options = {'name': 'pearson_baseline', 'user_based': False}
algo = KNNBasic(sim_options=sim_options)
dict_item_pearson = cross_validate(algo, data, measures=['rmse', 'mae'], cv=5, verbose=True)
print(dict_item_pearson)

plt.title('User based CF', fontsize=20)
plt.xlabel('Iteration of Cross-Validation', fontsize=15)
plt.ylabel('RMSE', fontsize=15)
plt.plot(dict_user_cosine['test_rmse'], 'r-')
plt.text(0, np.max(dict_user_cosine['test_rmse']), 'Cosine', color='red')
plt.plot(dict_user_msd['test_rmse'], 'g-')
plt.text(0, np.max(dict_user_msd['test_rmse']), 'MSD', color='green')
plt.plot(dict_user_pearson['test_rmse'], 'b-')
plt.text(0, np.max(dict_user_pearson['test_rmse']), 'Pearson', color='blue')
plt.show()

plt.title('Item based CF', fontsize=20)
plt.xlabel('Iteration of Cross-Validation', fontsize=15)
plt.ylabel('RMSE', fontsize=15)
plt.plot(dict_item_cosine['test_rmse'], 'r-')
plt.text(0, np.max(dict_item_cosine['test_rmse']), 'Cosine', color='red')
plt.plot(dict_item_msd['test_rmse'], 'g-')
plt.text(0, np.max(dict_item_msd['test_rmse']), 'MSD', color='green')
plt.plot(dict_item_pearson['test_rmse'], 'b-')
plt.text(0, np.max(dict_item_pearson['test_rmse']), 'Pearson', color='blue')
plt.show()

test_mean_rmse_list = []
for i in range(1, 101):
    algo = KNNBasic(k=i, verbose=False) # Default is User based
    dict_user_msd = cross_validate(algo, data, measures=['rmse'], cv=5, verbose=False)
    test_mean_rmse_list.append(dict_user_msd['test_rmse'].mean())
test_mean_rmse_array = np.array(test_mean_rmse_list)
plt.title('User based CF', fontsize=20)
plt.xlabel('Number of Neighbors (K)', fontsize=15)
plt.ylabel('RMSE', fontsize=15)
plt.plot(test_mean_rmse_array, '^')
plt.annotate('Best K = %s' % (np.argmin(test_mean_rmse_array)+1), 
             xy=(np.argmin(test_mean_rmse_array), np.min(test_mean_rmse_array)),
             fontsize=15,
             xytext=(np.argmin(test_mean_rmse_array)-0.1, np.min(test_mean_rmse_array)+0.1),
             arrowprops = dict(facecolor = 'green'))
plt.show()

test_mean_rmse_list = []
sim_options = {'user_based': False}
for i in range(1, 101):
    algo = KNNBasic(k=i, sim_options=sim_options, verbose=False)
    dict_item_msd = cross_validate(algo, data, measures=['rmse'], cv=5, verbose=False)
    test_mean_rmse_list.append(dict_item_msd['test_rmse'].mean())
test_mean_rmse_array = np.array(test_mean_rmse_list)
plt.title('Item based CF', fontsize=20)
plt.xlabel('Number of Neighbors (K)', fontsize=15)
plt.ylabel('RMSE', fontsize=15)
plt.plot(test_mean_rmse_array, '^')
plt.annotate('Best K = %s' % (np.argmin(test_mean_rmse_array)+1), 
             xy=(np.argmin(test_mean_rmse_array), np.min(test_mean_rmse_array)),
             fontsize=15,
             xytext=(np.argmin(test_mean_rmse_array)-0.1, np.min(test_mean_rmse_array)+0.1),
             arrowprops = dict(facecolor = 'green'))
plt.show()
