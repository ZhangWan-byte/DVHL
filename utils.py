import time
import copy
import random
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
import torch.optim as optim
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
# adapted from https://github.com/glanceable-io/ordinal-log-loss/blob/main/src/loss_functions.py
def ord_loss(logits, labels, alpha=1.5):
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

    err = -torch.log(1-probas)*abs(distances_tensor)**alpha
    loss = torch.sum(err,axis=1).mean()

    return loss


# https://github.com/WeiChengTseng/Pytorch-PCGrad/blob/master/pcgrad.py
class PCGrad():
    def __init__(self, optimizer, reduction='mean'):
        self._optim, self._reduction = optimizer, reduction
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def pc_backward(self, objectives):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        '''

        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).sum(dim=0)
        else: exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific 
        objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad


def kullback_leibler_loss(p, q, eps=1.0e-7):
    eps = torch.tensor(eps, dtype=p.dtype)
    kl_matr = torch.mul(p, torch.log(p + eps) - torch.log(q + eps))
    kl_matr.fill_diagonal_(0.)    
    return torch.sum(kl_matr)

def dist_mat_squared(x):
    batch_size = x.shape[0] 
    expanded = x.unsqueeze(1)
    tiled = torch.repeat_interleave(expanded, batch_size, dim=1)
    diffs = tiled - tiled.transpose(0, 1)
    sum_act = torch.sum(torch.pow(diffs,2), axis=2)    
    return sum_act

def norm_sym(x):
    # x.fill_diagonal_(0.)
    mask = torch.ones(x.shape).fill_diagonal_(0.).to(x.device)
    x = mask * x
    # original code is in-place operation -- not feasible for gradient backwards
    
    norm_facs = x.sum(axis=0, keepdim=True)
    x = x / norm_facs
    return 0.5*(x + x.t())

def calc_q(x, alpha):
    dists = dist_mat_squared(x)
    q = torch.pow((1 + dists/alpha), -(alpha+1)/2)
    
    q = norm_sym(q)

    q = q / q.sum()

    return q

def my_p_i(d, beta):
    x = - d * beta
    y = torch.exp(x)
    ysum = y.sum(dim=1, keepdim=True)
    return y / ysum

def calc_p(x, beta, perp=None):
    """calculate p_ij

    :param x: high dimensional (N, D)
    :param beta: variance in Gaussian to control neighbour range
    :param perp: perplexity, can be derived from beta
    :return: p_ij matrix in shape (N, N)
    """

    num_pts = x.shape[0]
    x = x.view(num_pts, -1)
    beta = beta.view(-1, 1)
    # k = min(num_pts - 1, int(3 * perp))
    k = np.rint(num_pts/2).astype(int)

    dists = torch.sqrt(dist_mat_squared(x))
    
    values, indices = torch.topk(dists, k, dim=1, largest=False)

    p_ij = my_p_i(values[:, 1:], beta)

    p_ij = p_ij / p_ij.sum()
    
    return p_ij