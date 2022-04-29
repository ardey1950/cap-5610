import pandas as pd #CSV file I/O, data processing
import numpy as np #large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
from sklearn.feature_extraction.text import TfidfVectorizer #Term Frequency - Inverse Document Frequency Vectorizer
from sklearn.metrics.pairwise import linear_kernel #Optimized Cosine Similarity computation
import time
import math

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

# Function that takes in an article as input and outputs a given number of recommended articles
def get_recommendations(article_id, num_recommended_articles=1, cosine_sim=cosine_sim):
    
    # Get the product_code from article_id
    product_code = article_id // 1000
    
    # Get the index of the product that matches the product code
    idx = indices[product_code]

    # Get the pairwsie similarity scores of all products with that product
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the products based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the score of the most similar products
    sim_scores = sim_scores[1:num_recommended_articles+1]

    # Get the product indices
    product_index = [i[0] for i in sim_scores]

    # Return a recommended article for the most similar product
    return articles_df[articles_df['product_code'].isin(articles_df2['product_code'].iloc[product_index].values)].head(num_recommended_articles)

IS_TEST = str(input())

# Compute predictions and post-process
list_customer_id = []
list_prediction = []
NUM_RECOMMENDATIONS_PER_CUSTOMER = 12

top_selling_articles_by_number_array = transactions_train_df.groupby('article_id').count()['t_dat'].sort_values(ascending=False).head(NUM_RECOMMENDATIONS_PER_CUSTOMER)

if IS_TEST.lower() == 'true':
    iterations = 0
    MAX_ITERATIONS_FOR_TEST = 10
    
for row in customers_df.itertuples():
    if IS_TEST.lower() == 'true':
        iterations += 1
        if iterations >= (MAX_ITERATIONS_FOR_TEST+1):
            break
    customer_id = getattr(row, 'customer_id')
    if customer_id in transactions_train_df.values:
        purchased_articles_array = transactions_train_df.loc[transactions_train_df['customer_id'] == customer_id, 'article_id'].values
        list_customer_id.append(customer_id)
        recommended_articles_string = ''
        for i in purchased_articles_array:
            list_of_articles_for_this_recommendation = (get_recommendations(i, NUM_RECOMMENDATIONS_PER_CUSTOMER)['article_id']).values.tolist()
            recommended_articles_string += ' '.join(str(j) for j in list_of_articles_for_this_recommendation)
            recommended_articles_string += str(' ')
        recommended_articles_list = recommended_articles_string.split()
        for i in purchased_articles_array:
            if i in recommended_articles_list:
                recommended_articles_list.remove(i)
        unique_recommended_articles = set(recommended_articles_list)
        list_prediction.append((' '.join(unique_recommended_articles))[:119]) # list is truncated to 12 articles each of length 9, space-separated, hence, 12*9+11
    else:
        list_customer_id.append(customer_id)
        recommended_articles_string = ''
        for i, curr_article_id in enumerate(top_selling_articles_by_number_array.index):
            recommended_articles_string += str(curr_article_id)
            recommended_articles_string += str(' ')  
        list_prediction.append(recommended_articles_string[:119]) # list is truncated to 12 articles each of length 9, space-separated, hence, 12*9+11

submission_content_based_detail_desc_df = pd.DataFrame(list(zip(list_customer_id, list_prediction)), columns =['customer_id', 'prediction'])
submission_content_based_detail_desc_df.to_csv('submission_content_based_detail_desc.csv', encoding='utf-8', index=False)
submission_content_based_detail_desc_df
