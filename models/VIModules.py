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


# grads = {}
# def save_grad(name):
#     def hook(grad):
#         grads[name] = grad
#     return hook
    

# borrowed from Hassan Askary
class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)

# borrowed from Hassan Askary
class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
            x = STEFunction.apply(x)
            return x


class VisualImitation(nn.Module):
    def __init__(self, size=1000, device=torch.device('cuda')):
        super(VisualImitation, self).__init__()

        self.size = size
        self.device = device
        self.ste = StraightThroughEstimator().to(self.device)


    def get_mask_mat(self, z, pts=0):
        """calculate mask matrix at pts for a datapoint

        :param z: [pos_x, pos_y]
        :param pts: reference grid point, defaults to 0
        :return: mask matrix at pts of datapoint z
        """

        a0 = torch.ones((z.shape[0], self.size, self.size), requires_grad=True).to(self.device)
        b0 = torch.ones((z.shape[0], self.size, self.size), requires_grad=True).to(self.device)
        a = a0 * z[:, 0].reshape(-1,1,1) * self.size
        b = b0 * z[:, 1].reshape(-1,1,1) * self.size

        # transpose happens here
        # pos - left-up are all positive
        gridx_pos = torch.arange(0,self.size).to(self.device)
        gridy_pos = torch.arange(0,self.size).to(self.device)

        x_pos = nn.ReLU()(a-gridx_pos)
        y_pos = nn.ReLU()(b-gridy_pos.reshape(-1,1))
        pos = nn.ReLU()(x_pos*y_pos)

        # neg - right-bottom are all positive
        gridx_neg = torch.arange(1,self.size+1).to(self.device)
        gridy_neg = torch.arange(1,self.size+1).to(self.device)

        x_neg = nn.ReLU()(gridx_neg-a)
        y_neg = nn.ReLU()(gridy_neg.reshape(-1,1)-b)
        neg = nn.ReLU()(x_neg*y_neg)

        # get mask_mat - only corresponding grid where zi falls into is positive, other grids are all zeros
        mask_mat = self.ste(pos*neg)

        return mask_mat


    def get_I_hat_single(self, z, size=1000):
        """calculate mask matrix of z

        :param z: [N, 3] - coordinates and data class
        :param size: size of I_hat, defaults to 1000
        :return: [N, 1000, 1000, 10] - for all N datapoints, calculate its single I_hat_i (I_hat_i[posx][posy]=one_hot_class_label)
        """
        # print("z: ", z.requires_grad)
        # z.register_hook(save_grad('z'))

        # get mask_mat
        mask_mat = self.get_mask_mat(z)

        # get masks
        masks = torch.stack([mask_mat]*10, dim=3)
        labels = F.one_hot(z[:, 2].long(), num_classes=10).float()

        # calculate final I_hat
        I_hat_single = masks * labels.unsqueeze(1).unsqueeze(1)
        I_hat_single = torch.transpose(I_hat_single, 1, 2).float()

        return I_hat_single


    def forward(self, z, labels):
        """calculate I_hat

        :param z: [N, 2] - coordinates
        :param labels: [N, 1] - data class
        """
        
        z = torch.hstack([z, labels.view(z.shape[0], 1)])

        I_hat_single = self.get_I_hat_single(z)
        # I_hat_single.register_hook(save_grad('I_hat_single'))
        
        I_hat, _ = torch.max(I_hat_single, dim=0)

        return I_hat.float()


def get_Ihat(Z, size=1000):
    mat = np.zeros((size,size))
    for i in range(len(Z)):
        xx = min(int(np.floor(Z[i,0]*size)), size-1)
        yy = min(int(np.floor(Z[i,1]*size)), size-1)
        if Z.shape[1]==3:
            mat[xx,yy] = Z[i, 2]
        else:
            mat[xx,yy] = 1
    return mat