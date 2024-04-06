import time
import copy
import numpy as np
import pandas as pd
import seaborn as sn
from tqdm import tqdm
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import sklearn
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from models import *
from utils import *
from datasets import *

from myPaCMAP import *


def gauss_clusters(
    n_clusters=10, dim=10, pts_cluster=100, random_state=None, cov=1, stepsize=1,
):

    if random_state is None:
        rng = np.random.RandomState()
    else:
        rng = random_state

    n = n_clusters * pts_cluster

    s = stepsize / np.sqrt(dim)
    means = np.linspace(np.zeros(dim), n_clusters * s, num=n_clusters, endpoint=False)
    cshift_mask = np.zeros(n_clusters, dtype=bool)
    cshift_mask[15] = True
    cov = np.eye(dim) * cov

    clusters = np.array(
        [rng.multivariate_normal(m, cov, size=(pts_cluster)) for m in means]
    )

    X = np.reshape(clusters, (-1, dim))

    y = np.repeat(np.arange(n_clusters), pts_cluster)
    return X, y

n_clusters = 20
dim = 50
pts_cluster = 1000
cluster_dist = 6
random_state = None

data, labels = gauss_clusters(
    n_clusters,
    dim,
    pts_cluster,
    stepsize=cluster_dist,
    random_state=random_state,
)

data = torch.from_numpy(data)
labels = torch.from_numpy(labels)

print(data.shape, labels.shape)






n_neighbors = 10
MN_ratio = 0.5
FP_ratio = 2.0

pair_neighbors = None
pair_MN = None
pair_FP = None

reducer = myPaCMAP(
    n_components=2, 
    n_neighbors=n_neighbors, 
    MN_ratio=MN_ratio, 
    FP_ratio=FP_ratio, 
    pair_neighbors=pair_neighbors, 
    pair_MN=pair_MN, 
    pair_FP=pair_FP, 
    verbose=True
)
print("reducer prepared")

z = reducer.fit_transform(
    data[:1000, :], 
    n_neighbors=np.ones((1000,), dtype=np.int32)*10, 
    n_MN=np.round(np.ones((1000,))*0.5).astype(np.int32), 
    n_FP=np.round(np.ones((1000,))*2.0).astype(np.int32)
)

print(z.shape)