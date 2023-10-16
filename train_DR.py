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

    

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='DR', help="train DR or HM")
    parser.add_argument('--MM_I_wPATH', type=str, default="./results/DR_weights.pt", help='weights to initialise DR model')
    parser.add_argument('--MM_II_wPATH', type=str, default=None, help='weights to initialise human model')
    parser.add_argument('--DR', type=str, default="UMAP", help='UMAP or t-SNE')
    
    # UMAP hyper-params
    parser.add_argument('--n_neighbors', type=int, default=15, help='n_neighbors for graph construction')
    parser.add_argument('--min_dist', type=int, default=0.1, help='UMAPLoss param')
    parser.add_argument('--negative_sample_rate', type=int, default=5, help='UMAPLoss param')
    parser.add_argument('--repulsion_strength', type=float, default=1.0, help='UMAPLoss param')
    
    # training params
    parser.add_argument('--device', type=str, default='cuda', help='device cpu or cuda')
    
    parser.add_argument('--batch_size_DR', type=int, default=1024, help='batch size - phase DR')
    parser.add_argument('--epochs_DR', type=int, default=20, help='training epochs - phase DR')

    parser.add_argument('--batch_size_HM', type=int, default=1, help='batch size - phase Human')
    parser.add_argument('--epochs_HM', type=int, default=100, help='training epochs - phase Human')

    # save path name
    parser.add_argument('--exp_name', type=str, help='name of this experiment')
    args = parser.parse_args()


    # create saving directory
    current_time = time.strftime('%y%m%d%H%M%S', time.localtime())
    result_path = "./results/{}_{}".format(current_time, args.exp_name)
    os.makedirs(result_path)
    print("saving dir: {}".format(result_path))
    with open(os.path.join(result_path, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=4)


    train_dataset, test_dataset = get_dataset(data='MNIST', DR='UMAP', args=args, ret='DR')

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
        pass
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
        device=torch.device(args.device)
    )


    # os.makedirs(os.path.join(result_path, "/phase1/"))
    # print("Go to {} and give feedback.".format(os.path.join(result_path, "phase1")))
    # input("Press Enter to continue...")


    # optimisation
    optimizer_DR = torch.optim.Adam(model.MM_I.parameters(), lr=1e-3)
    # scheduler_I = ...
    # optimizer_Human = torch.optim.Adam(model.MM_II.parameters(), lr=1e-3)
    # # scheduler_II = ...


    # # 1.1 human model
    # model = MMModel(
    #     MM_I_wPATH="./results/encoder_weights.pt", 
    #     MM_II_wPATH=None, 
    #     freeze=(True, False), 
    #     device=torch.device(args.device)
    # )
    # train_Human(model, criterion_Human, optimizer_Human, epochs=args.epochs_Human)

    # 1.2 DR model
    model, train_losses = train_epoch_DR(
        model=model, 
        criterion=criterion_DR, 
        optimizer=optimizer_DR, 
        train_dataset=train_dataset, 
        test_dataset=test_dataset, 
        epochs=args.epochs_DR
    )

    torch.save(model.state_dict(), os.path.join(result_path, 'DR_weights_{}.pt'.format(exp_name)))
    