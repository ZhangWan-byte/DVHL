# Implementation of trustworthiness and continuity (T&C), a quality measure for NLDR embeddings.
# For more details on the measure, see Venna, J., & Kaski, S. (2006). 
# Local multidimensional scaling. Neural Networks, 19(6-7), 889-899.
# This implementation has been written by Adrien Bibal (University of Namur).

import numpy as np

from scipy.spatial.distance import pdist, squareform

# This function computes the trustworthiness for a particular K, as in Venna et al.'s paper
def compute_trustworthiness(dataset, visu, projection_K, dataset_K, I_dataset):
	N   = len(visu)
	K   = len(projection_K[0])

	acc = 0
	for i in range(0, N):
		_, common_neighborhood, _ = np.intersect1d(projection_K[i, :], dataset_K[i, :], return_indices=True)
		UK_i = np.delete(projection_K[i, :], common_neighborhood)

		for j in UK_i:
			acc += np.where(I_dataset[i, :] == j)[0][0] - K

	return 1 - ((2/(N*K*((2*N)-(3*K)-1)))*acc)

# Compute T&C for all K in logarithmic scale, as performed by AUClogRNX,
# see Lee, J. A., Peluffo-Ordonez, D. H., & Verleysen, M. (2015). 
# Multi-scale similarities in stochastic neighbour embedding: Reducing dimensionality while preserving both local and global structure. Neurocomputing, 169, 246-261.
def compute(dataset, visu):
	N   = len(visu)

	D_dataset    = squareform(pdist(dataset))
	D_projection = squareform(pdist(visu))

	numerator   = 0.0
	denominator = 0.0

	I_dataset = np.argsort(D_dataset, 1)
	I_projection = np.argsort(D_projection, 1)

	# Remove the comparison of each point with itself. As the lists are sorted, O(n) in most of the cases.
	I_dataset_temp = []
	I_projection_temp = []
	for i in range(I_dataset.shape[0]):
		I_dataset_temp.append(np.delete(I_dataset[i, :], np.where(I_dataset[i, :] == i)[0]))
		I_projection_temp.append(np.delete(I_projection[i, :], np.where(I_projection[i, :] == i)[0]))
	
	I_dataset = np.array(I_dataset_temp)
	I_projection = np.array(I_projection_temp)

	# In the paper, no explicit constraints are put on the size of the neighborhood.
	# However, the two main equations restrict k < (2*N - 1)/3
	for k in range(1, int(((2*N)-1)/3)):
		projection_K = I_projection[:, :k]
		dataset_K = I_dataset[:, :k]

		# Trustworthiness and continuity are combined with a simple mean
		numerator += compute_trustworthiness(dataset, visu, projection_K, dataset_K, I_dataset)
		denominator += (1.0 / k)

	return numerator / denominator