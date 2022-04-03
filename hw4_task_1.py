import numpy as np #large, multi-dimensional arrays and matrices, along with high-level mathematical functions to operate on these arrays
import pandas as pd #data manipulation and analysis
from scipy.spatial.distance import cdist #compute distance between each pair of two collection of inputs
import copy #shallow and deep copy operations
import time #for time as a floating point number in seconds since epoch, in UTC
import random #pseudo-random numbers

#read the dataset comprising the data samples and ground-truth labels of all data samples
#header=None ensures the first row is not interpreted as header
data_df = pd.read_csv('./hw4_kmeans_data/data.csv', header=None)
label_df = pd.read_csv('./hw4_kmeans_data/label.csv', header=None)

#numpy representation of the dataframes
data = data_df.values
ground_truth_labels = label_df.values

#handling of harmless floating point warnings when computing 1-GeneralizedJaccard prediction accuracy
np.seterr(invalid='ignore')

#helper method - Euclidean distance between two data points
def compute_l2_dist(point1, point2):
    dist = 0
    for i in range(len(point1)):
        dist += (point2[i] - point1[i])**2
    return dist
        
#helper method - SSE for the computed clusters comprising the respective data points, centroids and predicted labels
def compute_sse(data, centroids, predicted_labels):
    sse = 0
    for i, x in enumerate(data):
        centroid = centroids[predicted_labels[i]]
        dist = compute_l2_dist(centroid, x)
        sse += dist
    return sse

#helper method - prediction accuracy using majority class vote labels of the data points per computed cluster
def compute_prediction_accuracy(k, ground_truth_labels, predicted_labels):
    matches = 0
    total = 0
    for idx in range(k):
        values, counts = np.unique(ground_truth_labels[predicted_labels==idx].transpose().flatten(), return_counts=True)
        if (len(counts) != 0):
            matches += np.max(counts)
            cluster_label = np.argmax(counts)
            print('Cluster label based on majority vote label of the data points in the cluster = %s' % (cluster_label))
        else:
            matches += 0
        total += np.sum(counts)
    print()
    return (matches/total)

supported_distance_metrices = np.array(['euclidean', 'cosine', 'jaccard'])

#randomly seed the generation of initial centroids
seed = random.randint(0, 65535)

'''
Implementation of the K-means algorithm from scratch
1 - Select K points as the initial centroids.
2 - repeat
3 -    Form K clusters by assigning all points to the closest centroid.
4 -    Recompute the centroid of each cluster
5 - until The centroids don't change

'''
def kmeans(data,
           k,
           distance_metric = 'euclidean',
           use_no_change_in_centroid_position=False,
           use_sse_increases_in_next_iteration=False,
           max_iterations=500):
    
    #record the invocation of the algorithm
    start_time = time.time()
    
    #throw an exception if the distance metric requested is not supported
    #only Eucliean, 1-Cosine and 1-GeneralizedJaccard are supported
    if ((distance_metric in supported_distance_metrices) == False):
        raise Exception('Provided distace metric is not supported.')
    
    #seed the generation of k initial centroids
    #note: the same initial centroids are used for a single run across all distance metrices, to enable comparison of performance across the 3 distance metrices
    np.random.seed(seed)
    initial_indices = np.random.choice(len(data), k, replace=False)
    
    #step 1 of the K-means algorithm
    initial_centroids = data[initial_indices, :]
    
    #step 3 of the K-means algorithm
    #scipy's cdist computes the distances between each pair of points in data and initial_centroids
    distances = cdist(data, initial_centroids, distance_metric)
    #form K clusters by assigning all points to the closest centroid
    predicted_labels = np.array([np.argmin(i) for i in distances])
    
    #variable initializations before the start of the repeat-until loop of the K-means algorithm
    current_sse = compute_sse(data, initial_centroids, predicted_labels) #SSE for the first set of k clusters
    current_centroids = copy.deepcopy(initial_centroids)
    num_iterations = 0   
    centroids_do_not_change = False
    
    #steps 2 - 5 of the K-means algorithm
    while (num_iterations < max_iterations):
        
        previous_centroids = copy.deepcopy(current_centroids)
        previous_sse = current_sse
        num_iterations += 1
        
        temp_centroids_list = []
        
        for idx in range(k):
            temp_centroid = data[predicted_labels==idx].mean(axis=0) #a recomputed centroid is the mean for a cluster
            temp_centroids_list.append(temp_centroid)
        
        current_centroids = np.vstack(temp_centroids_list)
        
        #step 4 of the K-means algorithm
        distances = cdist(data, current_centroids, distance_metric)
        predicted_labels = np.array([np.argmin(i) for i in distances])
        
        #SSE after the centroids are recomputed
        current_sse = compute_sse(data, current_centroids, predicted_labels)
        
        #checks if the centroids have changed since the last iteration, for convergence - tolerance is set to within 1%
        #note: use_no_change_in_centroid_position gives flexibility to the user whether to use this criteria
        if ((use_no_change_in_centroid_position == True) and (np.allclose(current_centroids, previous_centroids, rtol=0, atol=1e-02, equal_nan=True) == True)):
            centroids_do_not_change = True
            break
        
        #checks if the current SSE is more than the previous SSE - this should not happen - so, terminate the loop if this happens
        #note: use_sse_increases_in_next_iteration gives flexibility to the user whether to use this criteria
        if ((use_sse_increases_in_next_iteration == True) and (current_sse > previous_sse)):
            break
    
    #compute the prediction accuracy after the repeat-until loop has terminated
    prediction_accuracy = compute_prediction_accuracy(k, ground_truth_labels, predicted_labels)
    
    #time it took for the algoritrhm to run
    consumed_time = time.time() - start_time
    
    #returns a tuple containing the result parameters obtained from the execution of the K-means algorithm
    return distance_metric, num_iterations, initial_indices, centroids_do_not_change, current_sse, (current_sse - previous_sse), prediction_accuracy, consumed_time
 
#obtain the value of k to pass in to the algorithm
values, counts = np.unique(ground_truth_labels.transpose().flatten(), return_counts=True)
k = len(counts)

print('The value of k = %s' % (k))
print()

#following are various invocations of the K-means algorithm, to collect the data to answer the questions

'''
Q1: Run K-means clustering with Euclidean, Cosine and Jarcard similarity. Specify K= the 
number of categorical values of y (the number of classifications). Compare the SSEs of 
Euclidean-K-means, Cosine-K-means, Jarcard-K-means. Which method is better?

Q2: Compare the accuracies of Euclidean-K-means Cosine-K-means, Jarcard-K-means. First, 
label each cluster using the majority vote label of the data points in that cluster. Later, compute 
the predictive accuracy of Euclidean-K-means, Cosine-K-means, Jarcard-K-means. Which metric 
is better?
'''

distance_metric, num_iterations, initial_indices, centroids_do_not_change, current_sse, change_in_sse, prediction_accuracy, consumed_time = kmeans(data, k, max_iterations=20)
print('Initial centroid indices = %s' % (initial_indices))
print('Distance Metric = %s; Num Iterations = %s; Centroids Do Not Change = %s; Latest SSE = %s; Change in SSE = %s; Prediction Accuracy = %s; Processing Time = %s seconds' % (distance_metric, num_iterations, centroids_do_not_change, current_sse, change_in_sse, prediction_accuracy, consumed_time))
print()

distance_metric, num_iterations, initial_indices, centroids_do_not_change, current_sse, change_in_sse, prediction_accuracy, consumed_time = kmeans(data, k, distance_metric='cosine', max_iterations=20)
print('Initial centroid indices = %s' % (initial_indices))
print('Distance Metric = %s; Num Iterations = %s; Centroids Do Not Change = %s; Latest SSE = %s; Change in SSE = %s; Prediction Accuracy = %s; Processing Time = %s seconds' % (distance_metric, num_iterations, centroids_do_not_change, current_sse, change_in_sse, prediction_accuracy, consumed_time))
print()

distance_metric, num_iterations, initial_indices, centroids_do_not_change, current_sse, change_in_sse, prediction_accuracy, consumed_time = kmeans(data, k, distance_metric='jaccard', max_iterations=20)
print('Initial centroid indices = %s' % (initial_indices))
print('Distance Metric = %s; Num Iterations = %s; Centroids Do Not Change = %s; Latest SSE = %s; Change in SSE = %s; Prediction Accuracy = %s; Processing Time = %s seconds' % (distance_metric, num_iterations, centroids_do_not_change, current_sse, change_in_sse, prediction_accuracy, consumed_time))
print()

'''
Q3: Set up the same stop criteria: “when there is no change in centroid position OR when the 
SSE value increases in the next iteration OR when the maximum preset value (e.g., 500, you 
can set the preset value by yourself) of iteration is complete”, for Euclidean-K-means, Cosine-Kmeans, Jarcard-K-means. Which method requires more iterations and times to converge?

Q4: Compare the SSEs of Euclidean-K-means Cosine-K-means, Jarcard-K-means with respect to 
the following three terminating conditions: 
- when there is no change in centroid position
- when the SSE value increases in the next iteration
- when the maximum preset value (e.g., 100) of iteration is complete 
'''
distance_metric, num_iterations, initial_indices, centroids_do_not_change, current_sse, change_in_sse, prediction_accuracy, consumed_time = kmeans(data, k, use_no_change_in_centroid_position=True, use_sse_increases_in_next_iteration=True, max_iterations=100)
print('Initial centroid indices = %s' % (initial_indices))
print('Distance Metric = %s; Num Iterations = %s; Centroids Do Not Change = %s; Latest SSE = %s; Change in SSE = %s; Prediction Accuracy = %s; Processing Time = %s seconds' % (distance_metric, num_iterations, centroids_do_not_change, current_sse, change_in_sse, prediction_accuracy, consumed_time))
print()

distance_metric, num_iterations, initial_indices, centroids_do_not_change, current_sse, change_in_sse, prediction_accuracy, consumed_time = kmeans(data, k, distance_metric='cosine', use_no_change_in_centroid_position=True, use_sse_increases_in_next_iteration=True, max_iterations=100)
print('Initial centroid indices = %s' % (initial_indices))
print('Distance Metric = %s; Num Iterations = %s; Centroids Do Not Change = %s; Latest SSE = %s; Change in SSE = %s; Prediction Accuracy = %s; Processing Time = %s seconds' % (distance_metric, num_iterations, centroids_do_not_change, current_sse, change_in_sse, prediction_accuracy, consumed_time))
print()

distance_metric, num_iterations, initial_indices, centroids_do_not_change, current_sse, change_in_sse, prediction_accuracy, consumed_time = kmeans(data, k, distance_metric='jaccard', use_no_change_in_centroid_position=True, use_sse_increases_in_next_iteration=True, max_iterations=100)
print('Initial centroid indices = %s' % (initial_indices))
print('Distance Metric = %s; Num Iterations = %s; Centroids Do Not Change = %s; Latest SSE = %s; Change in SSE = %s; Prediction Accuracy = %s; Processing Time = %s seconds' % (distance_metric, num_iterations, centroids_do_not_change, current_sse, change_in_sse, prediction_accuracy, consumed_time))
print()