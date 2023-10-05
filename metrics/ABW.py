# Implementation of between-class average dist over within-class average dist (ABW, derived by Aupetit)
# For more details on the measure, see Lewis, J., Ackerman, M., & de Sa, V. (2012). 
# Human cluster evaluation and formal quality measures: A comparative study. In Proceedings of the Annual Meeting of the Cognitive Science Society (Vol. 34, No. 34).
# This implementation has been written by Adrien Bibal (University of Namur).

import numpy as np

from scipy.spatial.distance import pdist, squareform

# Between-class average distance (ABTN) and Within-class average distance (AWTN)
def ABTN_AWTN(visu, labels):
	n = len(visu)
	ABTN = [0.0, 0.0]
	AWTN = [0.0, 0.0]

	for i in range(n):
		for j in range(i+1, n):
			distance = pdist([visu[i], visu[j]])[0]
			if labels[i] == labels[j]:
				AWTN[0] += distance
				AWTN[1] += 1
			else:
				ABTN[0] += distance
				ABTN[1] += 1

	AWTN[0] = AWTN[0] / AWTN[1]
	ABTN[0] = ABTN[0] / ABTN[1]

	return ABTN[0], AWTN[0]
					

# Computed on the sum of all points, therefore depends on the number of instances.
def compute(visu, labels):
	ABTN, AWTN = ABTN_AWTN(visu, labels)
	return ABTN / AWTN
