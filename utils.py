import time
import copy
import numpy as np
import pandas as pd
import seaborn as sn
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits, fetch_openml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


def draw_Ihat(I_hat):
    """draw I_hat

    :param I_hat: grid-like img, e.g., (1000, 1000)
    """
    plt.figure(figsize=(16,16), dpi=1500)
    sn.heatmap(I_hat, cmap=sn.color_palette("rocket_r", as_cmap=True))
    plt.show()


def draw_z(z, cls, s=25, x_highlight=None, y_highlight=None, save_path=None, display=True, title=None):
    """draw data and labels

    :param z: (n, 2) -- 2D data
    :param cls: (n, ) -- label/class for datapoints
    :param s: size, defaults to 25
    """

    # re-organise as df
    try:
        tsne_data = np.hstack((z.numpy(), cls.reshape((-1,1)).numpy()))
    except:
        tsne_data = np.hstack((z, cls.reshape((-1,1))))

    tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))
    tsne_df = tsne_df.sort_values("label")

    # draw
    plt.figure(dpi=1500)
    
    # main figure
    sn.FacetGrid(tsne_df, hue="label", height=6, palette="Spectral").map(plt.scatter, "Dim_1", "Dim_2", s=s).add_legend()
    # highlights
    if x_highlight!=None and y_highlight!=None:
        plt.scatter(x_highlight, y_highlight, marker='*', s=20, c='black', label='Highlighted Point')

    # dpi    
    matplotlib.rcParams["figure.dpi"] = 1500
    
    # axes range
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    
    if save_path!=None:
        plt.savefig(save_path)

    if title!=None:
        plt.title(title)

    if display==True:
        plt.show()
    else:
        plt.ioff()
        plt.close()


def normalise_(zx, zy):
    """normalise coordinates to 0~1

    :param z: normalised coordinates
    """
    if type(zx) == type(torch.ones(1)):
        if torch.min(zx)!=torch.max(zx):
            zx = (zx - torch.min(zx)) / (torch.max(zx) - torch.min(zx))
        if torch.min(zy)!=torch.max(zy):
            zy = (zy - torch.min(zy)) / (torch.max(zy) - torch.min(zy))
        z = torch.hstack([zx.reshape(-1,1), zy.reshape(-1,1)])
    
    if type(zx) == type(np.ones(1)):
        if np.min(zx)!=np.max(zx):
            zx = (zx - np.min(zx)) / (np.max(zx) - np.min(zx))
        if np.min(zy)!=np.max(zy):
            zy = (zy - np.min(zy)) / (np.max(zy) - np.min(zy))
        z = np.hstack([zx.reshape(-1,1), zy.reshape(-1,1)])

    return z

def normalise(z):

    if len(z.shape)==2:
        z = normalise_(z[:,0], z[:,1])
    elif len(z.shape)==3:
        res_li = []
        for i in range(len(z)):
            res_li.append(normalise_(z[i,:,0].squeeze(), z[i,:,1].squeeze()))
        z = torch.stack(res_li, dim=0)
    else:
        print("wrong z.shape")
        exit()

    return z

def get_weights(x):
    x = x.view(-1)

    # neg = F.tanh(x) < 0
    # neg = neg.type(torch.int64)

    # pos = F.tanh(x) > 0
    # pos = pos.type(torch.int64)
    # pos *= -1
    # return neg+pos

    x = (x - torch.mean(x)) / torch.std(x)

    x = x * (-1)

    return x


def rotate_anticlockwise(z, times=1):

    R = torch.Tensor([[0, -1], 
                      [1, 0]])

    if type(z)==type(np.ones(1)):
        for i in range(times):
            z = np.dot(z, R)
    
    elif type(z)==type(torch.ones(1)):
        for i in range(times):
            z = torch.matmul(z, R)

    else:
        print("not implemented rotation")
        exit()

    z = normalise(z)

    return z


# ordinal loss
def ord_loss(logits, labels):
    num_classes = 5
    dist_matrix = torch.tensor([
        [0, 1, 2, 3, 4], 
        [4, 0, 1, 2, 3], 
        [3, 4, 0, 1, 2], 
        [2, 3, 4, 0, 1], 
        [4, 3, 2, 1, 0]
    ])

    # labels = torch.tensor([2, 3, 4, 4])

    # logits = torch.logit(torch.tensor([
    #     [0.1, 0.1, 0.6, 0.1, 0.1], 
    #     [0.1, 0.1, 0.6, 0.1, 0.1], 
    #     [0.1, 0.1, 0.6, 0.1, 0.1], 
    #     [0.025, 0.025, 0.025, 0.025, 0.9]
    # ]))

    probas = F.softmax(logits, dim=1).cuda()

    true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
    label_ids = len(labels)*[[k for k in range(num_classes)]]

    distances = [[dist_matrix[true_labels[j][i]][label_ids[j][i]]/np.sum([dist_matrix[n][label_ids[j][i]] for n in range(num_classes)]) for i in range(num_classes)] for j in range(len(labels))]
    distances_tensor = torch.tensor(distances,device='cuda:0', requires_grad=True)

    err = -torch.log(1-probas)*abs(distances_tensor)**1.5
    loss = torch.sum(err,axis=1).mean()

    return loss