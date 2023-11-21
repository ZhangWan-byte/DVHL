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


class myPCA(nn.Module):
    def __init__(self, n_components=2):
        super(myPCA, self).__init__()
        self.n_components = n_components

    def forward(self, X):

        n = X.shape[0]
        self.mean = torch.mean(X, axis=0)
        X = X - self.mean

        covariance_matrix = 1 / n * torch.matmul(X.T, X)
        eigenvalues, eigenvectors = torch.linalg.eig(covariance_matrix)
        eigenvalues = torch.norm(eigenvalues, dim=1)
        idx = torch.argsort(-eigenvalues)
        eigenvectors = eigenvectors[:, idx]
        self.proj_mat = eigenvectors[:, 0:self.n_components]

        return X.matmul(self.proj_mat)


class Encoder(nn.Module):

    def __init__ (self, output_dim=2, DR='UMAP'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128*6*6, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, output_dim)

        # t-SNE param
        if DR == 't-SNE':
            # self.alpha = nn.Parameter(torch.tensor([1.0])).to(device)
            # self.beta = nn.Parameter(torch.tensor([1.0])).to(device)
            self.alpha = torch.tensor(1.0, requires_grad=True, device="cuda")
            self.beta = torch.tensor(1.0, requires_grad=True, device="cuda")

            # self.alpha.requires_grad = True
            # self.beta.requires_grad = True
        else:
            self.alpha = None
            self.beta = None

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        return x

# model = Encoder(output_dim=2).cuda()
# print(model)
# print("num params: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))