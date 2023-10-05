# Implementation of MDS nonmetric stress, a quality measure for NLDR embeddings.
# For more details on the measure, see Kruskal, J. B. (1964). 
# Multidimensional scaling by optimizing goodness of fit to a nonmetric hypothesis. Psychometrika, 29(1), 1-27.
# This implementation has been written by Adrien Bibal (University of Namur).

import numpy as np

from scipy.spatial.distance import euclidean, pdist
from sklearn.preprocessing import scale
from sklearn.isotonic import IsotonicRegression

# Compute the non-metric stress
# data = high dimensional data
# visu = low dimensional data
# return the stress between the order of distances in data and the distances in visu
def compute(data, visu):
	data_dist = scale(pdist(data))
	visu_dist = scale(pdist(visu))

	f = IsotonicRegression()
	data_dist_hat = f.fit_transform(data_dist, visu_dist)

	numerator = np.sum([(visu_dist[i] - data_dist_hat[i])**2 for i in range(len(data_dist))])
	denominator = np.sum(visu_dist**2)

	stress = np.sqrt(numerator/denominator)

	return stress