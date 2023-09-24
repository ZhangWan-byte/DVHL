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


class VisualImitation_hard(nn.Module):
    def __init__(self, size=1000):
        super(VisualImitation_hard, self).__init__()

        self.size = size

    def row_idx(self, pos, size=1000):
        x, y = pos

        xx = min(int(np.floor(x*size)), size-1)
        yy = min(int(np.floor(y*size)), size-1)

        return xx*size + yy

    def forward(self, x):

        idx = torch.tensor([self.row_idx(pos=i, size=self.size) for i in x[:, :2].detach().numpy()], dtype=int).reshape(1, -1)
        src = x[:, 2].reshape(1, -1)

        I = torch.zeros(1, self.size*self.size, dtype=float, requires_grad=True)
        I1 = I.clone()
        I1 = I1.scatter_(dim=1, index=idx, src=src)
        I2 = I1.reshape((self.size, self.size))

        return I2


class VisualImitation1(nn.Module):
    def __init__(self, size=1000, device=torch.device('cuda')):
        super(VisualImitation1, self).__init__()

        self.size = size
        self.device = device


    def myact(self, x):
        """signum approximation

        :param x: input
        :return: output
        """
        return nn.Tanh()(x*100000)


    def get_mask_mat(self, z, pts=0):
        """calculate mask matrix at pts for a datapoint

        :param z: [pos_x, pos_y]
        :param pts: reference grid point, defaults to 0
        :return: mask matrix at pts of datapoint z
        """

        a0 = torch.ones((z.shape[0], 1000, 1000), requires_grad=True).to(self.device)
        a = a0 * z[:, 0].reshape(-1,1,1) * 1000
        b0 = torch.ones((z.shape[0], 1000, 1000), requires_grad=True).to(self.device)
        b = b0 * z[:, 1].reshape(-1,1,1) * 1000

        if pts==0:
            gridx = torch.arange(1,1001).to(self.device)
            gridy = torch.arange(1,1001).to(self.device)
        elif pts==1:
            gridx = torch.arange(0,1000).to(self.device)
            gridy = torch.arange(1,1001).to(self.device)
        elif pts==2:
            gridx = torch.arange(2,1002).to(self.device)
            gridy = torch.arange(2,1002).to(self.device)
        elif pts==3:
            gridx = torch.arange(1,1001).to(self.device)
            gridy = torch.arange(0,1000).to(self.device)

        x_new = self.myact(a-gridx)
        y_new = self.myact(b-gridy.reshape(-1,1))
        mask_mat = self.myact(x_new*y_new)

        return x_new, y_new, mask_mat


    def get_I_hat_single(self, z, size=1000):
        """calculate mask matrix of z

        :param z: [N, 3] - coordinates and data class
        :param size: size of I_hat, defaults to 1000
        :return: [N, 1000, 1000, 10] - for all N datapoints, calculate its single I_hat_i (I_hat_i[posx][posy]=one_hot_class_label)
        """
        # print("z: ", z.requires_grad)
        z.register_hook(save_grad('z'))
        # get mask_mat at point i
        _, _, mask1 = self.get_mask_mat(z=z[:, :2], pts=1)
        _, _, mask3 = self.get_mask_mat(z=z[:, :2], pts=3)
        print('sum of mask1 and mask3: ', torch.sum(mask1), torch.sum(mask3))
        mask1.register_hook(save_grad('mask1'))
        mask3.register_hook(save_grad('mask3'))

        # get the ultimate 0-1 mask
        mask = mask1 * mask3
        print('sum of mask: ', torch.sum(mask))
        mask.register_hook(save_grad('mask_0'))
        mask = (mask - 1) / 2 * (-1)
        mask.register_hook(save_grad('mask_1'))
        mask = torch.transpose(mask, 1, 2).float()
        # print(mask.shape)
        # print("mask: ", mask.requires_grad)
        mask.register_hook(save_grad('mask_2'))

        conv_weights = torch.Tensor([[[[0., 1., 0.],
                                       [1., 0., 1.],
                                       [0., 1., 0.]]]]).to(self.device)
        masks = F.conv2d(input=mask.unsqueeze(1), weight=conv_weights, bias=None, stride=1, padding=1)
        masks.register_hook(save_grad('masks_a'))
        # masks = torch.sigmoid((masks.squeeze()-3)*1000)
        # masks = torch.sigmoid((masks.view(z.size(0),1000,1000)-3)*1000)
        # masks = torch.relu((masks.view(z.size(0),1000,1000)-3))
        masks = torch.relu((masks.squeeze()-3))
        # print(masks.shape)
        # print("masks: ", masks.requires_grad)
        masks.register_hook(save_grad('masks_b'))

        masks = torch.stack([masks]*10, dim=3)
        labels = F.one_hot(z[:, 2].long(), num_classes=10).float()
        # print(masks.shape, labels.shape)

        I_hat_single = masks * labels.unsqueeze(1).unsqueeze(1)
        # print("I_hat_single: ", I_hat_single.requires_grad)

        return I_hat_single


    def forward(self, z):
        """calculate I_hat

        :param z: [N, 3] - coordinates and data class
        """
        # print("z: ", z.requires_grad)

        # I_hat_single: (N,1000,1000,10) I_hat_single matrix of all datapoints
        # I_hat_single = torch.vmap(self.get_I_hat_single)(z)
        I_hat_single = self.get_I_hat_single(z)
        I_hat_single.register_hook(save_grad('I_hat_single'))
        # I_hat = torch.any(I_hat_single, dim=0).float()
        # I_hat = torch.logical_or(I_hat_single, dim=0).float()
        I_hat, _ = torch.max(I_hat_single, dim=0)
        # print("I_hat: ", I_hat.requires_grad)

        return I_hat.float()