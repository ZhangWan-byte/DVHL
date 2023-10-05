# Implementation of LCMC, a quality measure for NLDR embeddings.
# For more details on the measure, see Chen, L., & Buja, A. (2009). 
# Local multidimensional scaling for nonlinear dimension reduction, graph drawing, and proximity analysis. Journal of the American Statistical Association, 104(485), 209-219.
# This implementation has been written by Adrien Bibal (University of Namur).

import numpy as np

from scipy.spatial.distance import pdist, squareform

# This function computes LCMC for a particular K, as in Chen et al.'s paper
def interesect_neighborhoods(dataset, visu, vK, nK):
	N   = len(visu)
	K   = len(vK[0])

	acc = sum(map(lambda vnK: np.intersect1d(vnK[0], vnK[1]).size, zip(vK, nK)))

	return acc - (K/(N-1))

# Compute LCMC for all K in logarithmic scale, as performed by AUClogRNX,
# see Lee, J. A., Peluffo-Ordonez, D. H., & Verleysen, M. (2015). 
# Multi-scale similarities in stochastic neighbour embedding: Reducing dimensionality while preserving both local and global structure. Neurocomputing, 169, 246-261.
def compute(dataset, visu):
	N   = len(visu)

	D_dataset    = squareform(pdist(dataset))
	D_projection = squareform(pdist(visu))

	numerator   = 0.0
	denominator = 0.0

	I_dataset = np.argsort(D_dataset, 1)[:, 1:]
	I_projection = np.argsort(D_projection, 1)[:, 1:]

	for i in range(1, N-1):
		vK = I_projection[:, :i]
		nK = I_dataset[:, :i]

		numerator += interesect_neighborhoods(dataset, visu, vK, nK)
		denominator += (1.0 / i)

	return numerator / denominator