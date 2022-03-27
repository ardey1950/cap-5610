import pandas as pd #CSV file I/O, data processing
import numpy as np #large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
from dask import dataframe as ds #Optimized CSV file I/O, data processing
from sklearn.feature_extraction.text import TfidfVectorizer #Term Frequency - Inverse Document Frequency Vectorizer
from sklearn.metrics.pairwise import linear_kernel #Optimized Cosine Similarity computation
import time

# Record start time before reading dataset
start = time.time()
articles_df = pd.read_csv('../h-and-m-personalized-fashion-recommendations/articles.csv')
articles_df2 = articles_df.drop_duplicates('product_code').reset_index(drop=True)
customers_df = pd.read_csv('../h-and-m-personalized-fashion-recommendations/customers.csv')
transactions_train_df = pd.read_csv('../h-and-m-personalized-fashion-recommendations/transactions_train.csv')
# Print time taken to read dataset
print("Time taken to read dataset: %s seconds" % (time.time() - start))

#Define a TF-IDF vectorizer object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english', max_df=0.80, min_df = 0.01, dtype=np.float32)

#Replace NaN with an empty string
articles_df2['detail_desc'] = articles_df2['detail_desc'].fillna('')

# Record start time before constructing TF-IDF matrix
# Construct the required TF-IDF matrix by fitting and transforming the data
start = time.time()
tfidf_matrix = tfidf.fit_transform(articles_df2['detail_desc'])
# Output the shape of tfidf_matrix
print(tfidf_matrix.shape)
# Print time taken to construct TF-IDF matrix
print("Time taken to construct TF-IDF matrix: %s seconds" % (time.time() - start))

# Print feature names
# print(tfidf.get_feature_names_out())

# Record start time before computing cosine similarity
start = time.time()
# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# Print time taken to compute cosine similarity
print("Time taken to compute cosine similarity matrix: %s seconds" % (time.time() - start))

# Construct a reverse map of indices and product codes
indices = pd.Series(articles_df2.index, index=articles_df2['product_code']).drop_duplicates()

# Function that takes in an article as input and outputs recommended articles
def get_recommendations(article_id, cosine_sim=cosine_sim):
    
    # Get the product_code from article_id
    product_code = article_id // 1000
    
    # Get the index of the product that matches the product code
    idx = indices[product_code]

    # Get the pairwsie similarity scores of all products with that product
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the products based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar products
    sim_scores = sim_scores[1:11]

    # Get the product indices
    product_indices = [i[0] for i in sim_scores]

    # Return recommended articles for the top 10 products
    return articles_df[articles_df['product_code'].isin(articles_df2['product_code'].iloc[product_indices].values)]

# Print example recommendation
print(get_recommendations(108775015))
