# Implementation of Curvilinear Component Analysis (CCA), a quality measure for NLDR embeddings.
# For more details on the measure, see Demartines, P., & Hérault, J. (1997). 
# Curvilinear component analysis: A self-organizing neural network for nonlinear mapping of data sets. IEEE Transactions on neural networks, 8(1), 148-154.
# This implementation has been written by Adrien Bibal (University of Namur).

from math import exp
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.preprocessing import scale


# Compute the sigmoid function on x
def sigmoid(x):
	return 1 / (1 + exp(-x))

# Compute the stress function of CCA (the focus of this stress is on the output space)
# data = high dimensional data
# visu = low dimensional data
# return the CCA stress between data and visu
def compute(data, visu):
	# DV = Distance Vector
	DV1 = scale(pdist(data, 'euclidean'))
	DV2 = scale(pdist(visu, 'euclidean'))

	stress = 0
	for i in range(len(data)):
		stress += ((DV1[i] - DV2[i])**2)*(1 - sigmoid(DV2[i]))

	return stress