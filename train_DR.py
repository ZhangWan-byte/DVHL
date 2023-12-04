import os
import time
import copy
import json
import argparse
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

import torchvision as tv
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.utils import save_image

from models import *
from utils import *
from datasets import *

from train_epoch import train_epoch_DR


if __name__=='__main__':

    
    # , default="./data/pretrain_results/DR_weights.pt"
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='DR', help="train DR or HM")
    parser.add_argument('--MM_I_wPATH', type=str, \
        help='weights to initialise DR model')
    parser.add_argument('--MM_II_wPATH', type=str, default="./data/pretrain_results/HM_weights.pt", \
        help='weights to initialise human model')
    parser.add_argument('--dataset', type=str, default="MNIST", help='MNIST, COIL-20, etc')
    parser.add_argument('--DR', type=str, default="UMAP", help='UMAP or t-SNE')
    
    # UMAP hyper-params
    parser.add_argument('--n_neighbors', type=int, default=15, help='n_neighbors for graph construction')
    parser.add_argument('--min_dist', type=int, default=0.1, help='UMAPLoss param')
    parser.add_argument('--negative_sample_rate', type=int, default=5, help='UMAPLoss param')
    parser.add_argument('--repulsion_strength', type=float, default=1.0, help='UMAPLoss param')
    
    # training params
    parser.add_argument('--device', type=str, default='cuda', help='device cpu or cuda')
    parser.add_argument('--batch_size_DR', type=int, default=1000, help='batch size - phase DR')
    parser.add_argument('--epochs_DR', type=int, default=20, help='training epochs - phase DR')
    parser.add_argument('--gamma', type=float, default=0.3, help='gamma * DR_loss + (1-gamma) * HM_loss')

    # data
    parser.add_argument('--feedback_path', type=str, default="./results/231016000038_I_0/feedback.pt", \
        help='path of human feedback')

    # save path name
    parser.add_argument('--exp_dir', type=str, default=None, help='directory of this n-phase exp, e.g., 231024172445')
    parser.add_argument('--exp_name', type=str, help='name of this experiment, e.g., I-1')
    
    args = parser.parse_args()


    # create saving directory
    if args.exp_dir==None:
        current_time = time.strftime('%y%m%d%H%M%S', time.localtime())
        result_path = "./results/{}/{}/".format(current_time, args.exp_name)
    else:
        result_path = "./results/{}/{}/".format(args.exp_dir, args.exp_name)

    os.makedirs(result_path, exist_ok=True)
    print("saving dir: {}".format(result_path))
    with open(os.path.join(result_path, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=4)


    train_dataset, test_dataset = get_dataset(args=args, data=args.dataset, DR=args.DR)

    # loss function
    if args.DR=="UMAP":
        criterion_DR = UMAPLoss(
            device=args.device, 
            min_dist=args.min_dist, 
            batch_size=args.batch_size_DR, 
            negative_sample_rate=args.negative_sample_rate, 
            edge_weight=None, 
            repulsion_strength=args.repulsion_strength
        )
    elif args.DR=="t-SNE":
        criterion_DR = kullback_leibler_loss
    else:
        print("wrong args.DR")
        exit()


    # 0. model initialisation
    model = MMModel(
        MM_I_wPATH=args.MM_I_wPATH, 
        MM_II_wPATH=args.MM_II_wPATH, 
        cnn_layers=[1,1,1,1], 
        VI_size=100, 
        freeze=(False, True), 
        batch_size=args.batch_size_DR, 
        device=torch.device(args.device), 
        DR=args.DR
    )


    # optimisation
    # optimizer_DR = PCGrad(torch.optim.Adam(model.MM_I.parameters(), lr=1e-4))
    optimizer_DR = torch.optim.Adam([
        {"params": model.MM_I.parameters(), "lr": 1e-4}, 
        {"params": model.beta, "lr": 1.0}
    ])
    # scheduler_DR =torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_DR, T_0=20, eta_min=1e-8)
    scheduler_DR = None

    # 1.2 DR model
    model, train_losses, eval_losses = train_epoch_DR(
        args=args, 
        model=model, 
        criterion=criterion_DR, 
        optimizer=optimizer_DR, 
        scheduler=scheduler_DR, 
        train_dataset=train_dataset, 
        test_dataset=test_dataset, 
        epochs=args.epochs_DR, 
        result_path=result_path, 
        gamma=args.gamma
    )

    # torch.save(model.MM_I.state_dict(), os.path.join(result_path, 'DR_weights_{}.pt'.format(args.exp_name)))
    torch.save(torch.tensor(train_losses), os.path.join(result_path, 'train_losses_{}.pt'.format(args.exp_name)))
    torch.save(torch.tensor(eval_losses), os.path.join(result_path, 'eval_losses_{}.pt'.format(args.exp_name)))
    