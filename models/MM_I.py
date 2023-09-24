import time
import copy
import numpy as np
import pandas as pd
import seaborn as sn
from tqdm import tqdm
import matplotlib.pyplot as plt

import sklearn
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class PCA(nn.Module):
    def __init__(self, n_components=2):
        super(PCA, self).__init__()
        self.n_components = n_components

    def forward(self, X):

        n = X.shape[0]
        self.mean = torch.mean(X, axis=0)
        X = X - self.mean

        covariance_matrix = 1 / n * torch.matmul(X.T, X)
        eigenvalues, eigenvectors = torch.eig(covariance_matrix, eigenvectors=True)
        eigenvalues = torch.norm(eigenvalues, dim=1)
        idx = torch.argsort(-eigenvalues)
        eigenvectors = eigenvectors[:, idx]
        self.proj_mat = eigenvectors[:, 0:self.n_components]

        return X.matmul(self.proj_mat)