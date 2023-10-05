# Implementation of CAL, a quality measure for NLDR embeddings.
# For more details on the measure, see  Caliński, T., & Harabasz, J. (1974). 
# A dendrite method for cluster analysis. Communications in Statistics-theory and Methods, 3(1), 1-27.
# This implementation has been written by Adrien Bibal (University of Namur).

import numpy as np

from scipy.spatial.distance import pdist, squareform

# Returns:
# 	d_g : Mean of the general distance between all points
#	d_k : Mean of the distances between all points within a cluster
def distances(visu, labels):
	test = {}
	# Init
	n = len(visu)
	d_g = [0.0, 0.0] # (sum of the distances, counter of instances used to compute the sum)
	d_k = {}
	for label in np.unique(labels):
		test[label] = []
		d_k[label] = [0.0, 0.0] # (sum of the distances, counter of instances used to compute the sum)

	# Compute the distances (general distance and the distances for all labels/clusters) 
	for i in range(n):
		for j in range(i+1, n):
			distance = pdist([visu[i], visu[j]], 'sqeuclidean')[0]
			if labels[i] == labels[j]:
				d_k[labels[i]][0] += distance
				d_k[labels[i]][1] += 1
				test[labels[i]].append(distance)
			
			d_g[0] += distance
			d_g[1] += 1

	for label in np.unique(labels):
		d_k[label][0] = d_k[label][0] / d_k[label][1]
		test[label] = np.mean(test[label])
	
	d_g[0] = d_g[0] / d_g[1]

	return d_g, d_k

def Ak(visu, labels, d_g, d_k):
	n = len(visu)
	k = len(np.unique(labels))
	acc = 0

	for label in np.unique(labels):
		acc += (d_g[0] - d_k[label][0])*(d_k[label][1] - 1)

	return acc / (n - k)

# Compute CAL, also called VRS in Calinski's paper
def compute(visu, labels):
	n = float(len(visu))
	k = float(len(np.unique(labels)))

	d_g, d_k = distances(visu, labels)

	A_k = Ak(visu, labels, d_g, d_k)

	numerator = d_g[0] + ( ((n-k)/(k-1)) * A_k )
	denominator = d_g[0] - A_k

	return numerator / denominator
