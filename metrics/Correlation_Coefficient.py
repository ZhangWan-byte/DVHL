# Implementation of the correlation coefficient, a quality measure for NLDR embeddings.
# For more details on the measure, see Geng, X., Zhan, D. C., & Zhou, Z. H. (2005). 
# Supervised nonlinear dimensionality reduction for visualization and classification. IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), 35(6), 1098-1107.
# This implementation has been written by Adrien Bibal (University of Namur).

import numpy as np
from scipy.spatial.distance import pdist

# Compute the correlation coef between the distance vector of each visu
# data = high dimensional data
# visu = low dimensional data
# return the coorelation between the pairwise distances in data and visu
def compute(data, visu):
	# DV = Distance Vector
	DV1 = pdist(data, 'euclidean')
	DV2 = pdist(visu, 'euclidean')

	return np.corrcoef(DV1, DV2)[0,1]