# Implementation of Neighbor Retrieval Visualizer (NeRV), a quality measure for NLDR embeddings.
# For more details on the measure, see Venna, J., Peltonen, J., Nybo, K., Aidos, H., & Kaski, S. (2010). 
# Information retrieval perspective to nonlinear dimensionality reduction for data visualization. Journal of Machine Learning Research, 11(Feb), 451-490.
# This implementation has been written by Adrien Bibal (University of Namur).

import numpy as np

from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import scale
from scipy.stats import entropy as H

# This function is from Laurens van der Maaten (https://lvdmaaten.github.io/tsne/)
def Hbeta(D, beta = 1.0):
	"""Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""
	
	# Compute P-row and corresponding perplexity
	P = np.exp(-D.copy() * beta);
	sumP = sum(P);
	H = np.log(sumP) + beta * np.sum(D * P) / sumP;
	P = P / sumP;
	return H, P;

# This function is from Laurens van der Maaten (https://lvdmaaten.github.io/tsne/)
# Compute the value of sigma for a given perplexity
def x2p(D, tol = 1e-5, perplexity = 30.0):
	"""Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""

	# Adrien: Removed these lines because the distances are already given as input (D)
	# # Initialize some variables
	# print("Computing pairwise distances...")
	# (n, d) = X.shape;
	# sum_X = np.sum(np.square(X), 1);
	# D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X);

	n = len(D)

	P = np.zeros((n, n));
	beta = np.ones((n, 1));
	logU = np.log(perplexity);
    
	# Loop over all datapoints
	for i in range(n):
	
		# Compute the Gaussian kernel and entropy for the current precision
		betamin = -np.inf; 
		betamax =  np.inf;
		Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))];
		(H, thisP) = Hbeta(Di, beta[i]);
			
		# Evaluate whether the perplexity is within tolerance
		Hdiff = H - logU;
		tries = 0;
		while np.abs(Hdiff) > tol and tries < 50:
				
			# If not, increase or decrease precision
			if Hdiff > 0:
				betamin = beta[i];
				if betamax == np.inf or betamax == -np.inf:
					beta[i] = beta[i] * 2;
				else:
					beta[i] = (beta[i] + betamax) / 2;
			else:
				betamax = beta[i];
				if betamin == np.inf or betamin == -np.inf:
					beta[i] = beta[i] / 2;
				else:
					beta[i] = (beta[i] + betamin) / 2;
			
			# Recompute the values
			(H, thisP) = Hbeta(Di, beta[i]);
			Hdiff = H - logU;
			tries = tries + 1;
			
	# print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))
	return np.sqrt(1 / beta)

def r(distances, j, i, sigma):
	numerator = np.exp(-1 * (distances[j, i]**2)) / (sigma**2)
	
	denominator = 0
	for k in range(len(distances)):
		denominator += np.exp(-1*(distances[k, i]**2)) / (sigma**2)

	return numerator / denominator
	
def compute(data, visu, l=0.5):
	data_distances = pairwise_distances(data, metric='euclidean')
	data_distances = data_distances / (np.sum(data_distances) / 2) # Divided by two because we counted the distances twice (symetric matrix)
	sigma_p = x2p(data_distances, perplexity=5)[:, 0] # Compute sigma for each point in the original data

	visu_distances = pairwise_distances(visu, metric='euclidean')
	visu_distances = visu_distances / (np.sum(visu_distances) / 2) # Divided by two because we counted the distances twice (symetric matrix)
	sigma_q = x2p(visu_distances, perplexity=5)[:, 0] # Compute sigma for each point in the projection

	left = 0
	right = 0
	for i in range(len(data)): # Sum on all i for the mean
		for j in range(len(data)):
			if i != j:
				p = r(data_distances, j, i, sigma_p[i]) # Compute p(j|i)
				q = r(visu_distances, j, i, sigma_q[i]) # Compute q(j|i)
				
				left += p*np.log(p/q) # Compute KL(p||q)
				right += q*np.log(q/p) # Compute KL(q||p)

	return ((l*left) + ((1-l)*right))