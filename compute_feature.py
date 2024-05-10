import os
import rpy2.robjects as robjects

import numba
import numpy as np
from scipy.spatial.distance import pdist, squareform

from sklearn.metrics import silhouette_score

import networkx as nx

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

from sklearn.metrics import mean_squared_error, silhouette_score

from utils import normalise


# 1. AUClogRNX

@numba.njit("f4(f4[:,:],f4[:,:],i4[:,:],i4[:,:])", cache=True)
def QNX_K(dataset, visu, vK, nK):
    N = len(visu)
    K = len(vK[0])

    # acc = sum(map(lambda vnK: np.intersect1d(vnK[0], vnK[1]).size, zip(vK, nK)))

    # print(np.intersect1d(vK, nK))

    acc = 0 #[]#np.array([])
    for i in numba.prange(len(vK)):
        acc += np.intersect1d(vK[i], nK[i]).size

    return acc / (K * N)

@numba.njit("f4(f4[:,:],f4[:,:],i4[:,:],i4[:,:])", cache=True)
def RNX_K(dataset, visu, vK, nK):
    N = len(visu)
    K = len(vK[0])
    # print(dataset.shape, visu.shape, vK.shape, nK.shape)
    # print(dataset.dtype, visu.dtype, vK.dtype, nK.dtype)
    QNX = QNX_K(dataset, visu, vK, nK)

    numerator = ((N - 1) * QNX) - K
    denominator = N - 1 - K

    return numerator / denominator

@numba.njit("f4(f4[:,:],f4[:,:],i4[:,:],i4[:,:])", parallel=True, nogil=True, cache=True)
def logRNX(dataset, visu, I_dataset, I_projection):
    N = len(visu)

    # D_dataset = squareform(pdist(dataset))
    # D_projection = squareform(pdist(visu))

    numerator = 0.0
    denominator = 0.0

    # I_dataset = np.argsort(D_dataset, 1)[:, 1:].astype('int32')
    # I_projection = np.argsort(D_projection, 1)[:, 1:].astype('int32')

    for k in numba.prange(1, N - 1):
        vK = I_projection[:, :k]
        nK = I_dataset[:, :k]

        numerator += (RNX_K(dataset, visu, vK, nK) / k)
        denominator += (1.0 / k)

    return numerator / denominator


def compute_AUClogRNX(data, visu):
    """ Compute AUClogRNX
    Inputs
    ------
    data = high dimensional data
    visu = low dimensional data

    Output
    ------
    return the log of the AUC of K neighborhoods for a growing K
    """

    D_dataset = squareform(pdist(data)).astype('float32')
    D_projection = squareform(pdist(visu)).astype('float32')

    # print(D_dataset.shape, D_dataset.dtype, D_projection.shape, D_projection.dtype)

    I_dataset = np.argsort(D_dataset, 1)[:, 1:].astype('int32')
    I_projection = np.argsort(D_projection, 1)[:, 1:].astype('int32')

    return logRNX(data, visu, I_dataset, I_projection)

# 2. Scagnostics

def compute_scagnostics(x, y):
    # print(os.getcwd())
    all_scags = {}
    r_source = robjects.r['source']
    # r_source(os.path.join(path, '../../DRflow/metrics/get_scag.r'))
    r_source(os.path.join('./metrics/get_scag.r'))
    # print(os.path.join('./metrics/get_scag.r'))
    r_getname = robjects.globalenv['scags']
    scags = r_getname(robjects.FloatVector(x), robjects.FloatVector(y))
    all_scags['outlying'] = scags[0]
    all_scags['skewed'] = scags[1]
    all_scags['clumpy'] = scags[2]
    all_scags['sparse'] = scags[3]
    all_scags['striated'] = scags[4]
    all_scags['convex'] = scags[5]
    all_scags['skinny'] = scags[6]
    all_scags['stringy'] = scags[7]
    all_scags['monotonic'] = scags[8]
    return all_scags

# 3. DSC

def get_centroids(visu, labels):
    centroids = {}
    counter = {}

    for index in range(len(visu)):
        if labels[index] in centroids:
            centroids[labels[index]] = (
                centroids[labels[index]][0] + visu[index][0], centroids[labels[index]][1] + visu[index][1])
            counter[labels[index]] += 1.0
        else:
            centroids[labels[index]] = (visu[index][0], visu[index][1])
            counter[labels[index]] = 1.0

    for label in centroids.keys():
        centroids[label] = (centroids[label][0] / counter[label], centroids[label][1] / counter[label])

    return centroids

def compute_dsc(visu, labels):
    centroids = get_centroids(visu, labels)

    misclassified_counter = 0.0
    for index in range(len(visu)):
        distance_centroid = pdist([visu[index], centroids[labels[index]]])
        min_distance = distance_centroid  # init
        for label in centroids.keys():
            distance = pdist([visu[index], centroids[label]])
            if distance < min_distance:
                min_distance = distance

        if min_distance < distance_centroid:  # If the minimal distance is not the distance between the point and the centroid of its class
            misclassified_counter += 1.0

    return misclassified_counter / len(visu)

# 4. SC
def compute_silhouette(visu, labels):
	return silhouette_score(visu, labels)

# 5. MST length
def MST_length(z, k=5):
    z = normalise(z)

    A = kneighbors_graph(z, n_neighbors=k, mode='distance', metric='euclidean', include_self=False)

    G = nx.Graph(A)

    MST = nx.minimum_spanning_tree(G)

    length = sum(weight for _, _, weight in MST.edges(data='weight'))

    return length

# 6. ph dim

def sample_W(W, nSamples, isRandom=True):
    n = W.shape[0]
    random_indices = np.random.choice(n, size=nSamples, replace=False)
    return W[random_indices]

def calculate_ph_dim(W, min_points=200, max_points=1000, point_jump=50,  
        h_dim=0, print_error=False):
    from ripser import ripser
    # sample_fn should output a [num_points, dim] array
    
    # sample our points
    test_n = range(min_points, max_points, point_jump)
    lengths = []
    for n in test_n:
        diagrams = ripser(sample_W(W, n))['dgms']
        
        if len(diagrams) > h_dim:
            d = diagrams[h_dim]
            d = d[d[:, 1] < np.inf]
            lengths.append((d[:, 1] - d[:, 0]).sum())
        else:
            lengths.append(0.0)
    lengths = np.array(lengths)
    
    # compute our ph dim by running a linear least squares
    x = np.log(np.array(list(test_n)))
    y = np.log(lengths)
    N = len(x)
    m = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
    b = y.mean() - m * x.mean()
    
    error = ((y - (m * x + b)) ** 2).mean()
    
    if print_error:
        print(f"Ph Dimension Calculation has an approximate error of: {error}.")
    return 1 / (1 - m)


# final func

def generate_features(z, labels, data, use_auclogrnx=True):
    # 1. scagnostics
    scag = compute_scagnostics(z[:,0], z[:,1])

    # 2. separability
    dsc = compute_dsc(z, labels)
    sc = compute_silhouette(z, labels)

    # 3. accuracy
    if use_auclogrnx:
        auclogrnx = compute_AUClogRNX(data=data.astype('float32'), visu=z.astype('float32'))

    # 4. extra
    mst_length = MST_length(z)
    phd = calculate_ph_dim(z)

    # feature
    feats = []
    feats.extend([scag[k] for k in sorted(scag.keys())])
    feats.append(dsc)
    feats.append(sc)
    if use_auclogrnx:
        feats.append(auclogrnx)
    feats.append(mst_length)
    feats.append(phd)

    return np.array(feats)