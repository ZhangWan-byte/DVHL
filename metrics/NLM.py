# Implementation of Sammon's nonlinear mapping stress, a quality measure for NLDR embeddings.
# For more details on the measure, see Sammon, J. W. (1969). 
# A nonlinear mapping for data structure analysis. IEEE Transactions on computers, 100(5), 401-409.
# This implementation has been written by Adrien Bibal (University of Namur).

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.preprocessing import scale

# Compute the stress function of NLM (the focus of this stress is on the input space)
# data = high dimensional data
# visu = low dimensional data
# return the NLM stress between data and visu
def compute(data, visu):
	# DV = Distance Vector
	DV1 = scale(pdist(data, 'euclidean'))
	DV2 = scale(pdist(visu, 'euclidean'))

	stress = 0
	for i in range(len(DV1)):
		stress += ((DV1[i] - DV2[i])**2) / DV1[i]

	stress *= 1 / np.sum(DV1)

	return stress