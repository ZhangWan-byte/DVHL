# Implementation of hypothesis-margin HM, a quality measure for NLDR embeddings.
# For more details on the measure, see Gilad-Bachrach, R., Navot, A., & Tishby, N. (2004). 
# Margin based feature selection-theory and algorithms. In Proceedings of ICML (p. 43).
# This implementation has been written by Adrien Bibal (University of Namur).

import numpy as np

from scipy.spatial.distance import pdist, squareform

# Return a tuple composed of the nearhit (the closest point to x that has the same class as x)
# and the nearmiss (the closest point to x which has another class than x)
def nearhit_nearmiss(x, x_label, visu, labels):
	first_nearhit = True
	first_nearmiss = True

	# Compute nearhit and nearmiss
	for index in range(len(visu)):
		if (visu[index] != x).any():
			distance = pdist([visu[index], x])
			if labels[index] == x_label and (first_nearhit or nearhit_distance > distance):
				nearhit_distance = distance
				nearhit = visu[index]
				first_nearhit = False
			elif labels[index] != x_label and (first_nearmiss or nearmiss_distance > distance):
				nearmiss_distance = distance
				nearmiss = visu[index]
				first_nearmiss = False

	return nearhit, nearmiss

# Computed on the sum of all points, therefore depends on the number of instances.
# In order to be comparable amongs different visualizations, the result is divided by the number of points.
def compute(visu, labels):
	HM = 0.0
	for index in range(len(visu)):
		nearhit, nearmiss = nearhit_nearmiss(visu[index], labels[index], visu, labels)
		HM += pdist([visu[index], nearmiss])[0] - pdist([visu[index], nearhit])[0]

	return HM / len(visu)