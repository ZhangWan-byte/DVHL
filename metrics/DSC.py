# Implementation of the DSC, a quality measure for NLDR embeddings.
# For more details on the measure, see Sips, M., Neubert, B., Lewis, J. P., & Hanrahan, P. (2009, June). 
# Selecting good views of high‐dimensional data using class consistency. In Computer Graphics Forum (Vol. 28, No. 3, pp. 831-838)
# This implementation has been written by Adrien Bibal (University of Namur).

import numpy as np
from scipy.spatial.distance import pdist, squareform


# Compute the centroid of each label
def get_centroids(visu, labels):
    centroids = {}
    counter = {}

    for index in range(len(visu)):
        if labels[index] in centroids:
            centroids[labels[index]] = (
                centroids[labels[index]][0] + visu[index][0], centroids[labels[index]][1] + visu[index][1])
            counter[labels[index]] += 1.0
        else:
            centroids[labels[index]] = (visu[index][0], visu[index][1])
            counter[labels[index]] = 1.0

    for label in centroids.keys():
        centroids[label] = (centroids[label][0] / counter[label], centroids[label][1] / counter[label])

    return centroids


# Compute DSC
# In order to be comparable amongs different visualizations, the result is divided by the number of points.
def compute(visu, labels):
    centroids = get_centroids(visu, labels)

    misclassified_counter = 0.0
    for index in range(len(visu)):
        distance_centroid = pdist([visu[index], centroids[labels[index]]])
        min_distance = distance_centroid  # init
        for label in centroids.keys():
            distance = pdist([visu[index], centroids[label]])
            if distance < min_distance:
                min_distance = distance

        if min_distance < distance_centroid:  # If the minimal distance is not the distance between the point and the centroid of its class
            misclassified_counter += 1.0

    return misclassified_counter / len(visu)
