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



if __name__=='__main__':

    

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--MM_I_wPATH', type=str, default="./results/encoder_weights.pt", help='weights to initialise DR model')
    parser.add_argument('--MM_II_wPATH', type=str, default=None, help='weights to initialise human model')
    parser.add_argument('--DR', type=str, default="UMAP", help='config file path')
    
    # UMAP hyper-params
    parser.add_argument('--n_neighbors', type=int, default=15, help='n_neighbors for graph construction')
    parser.add_argument('--min_dist', type=int, default=0.1, help='UMAPLoss param')
    parser.add_argument('--negative_sample_rate', type=int, default=5, help='UMAPLoss param')
    parser.add_argument('--repulsion_strength', type=float, default=1.0, help='UMAPLoss param')
    
    # training params
    parser.add_argument('--device', type=str, default='cuda', help='device')
    
    parser.add_argument('--batch_size_DR', type=int, default=1024, help='batch size - phase _DR')
    parser.add_argument('--epochs_DR', type=int, default=20, help='training epochs - phase _DR')

    parser.add_argument('--batch_size_Human', type=int, default=1, help='batch size - phase Human')
    parser.add_argument('--epochs_Human', type=int, default=100, help='training epochs - phase Human')
    
    args = parser.parse_args()


    # create saving directory
    current_time = time.strftime('%m%d%H%M%S', time.localtime())
    result_path = "./results/{}".format(current_time)
    os.makedirs(result_path)
    print("saving dir: {}".format(result_path))
    with open(result_path, 'w') as f:
        json.dump(args.__dict__, f, indent=4)


    # data preprocessing
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainTransform  = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Normalize((0.1307,), (0.3081,))])
    trainset = tv.datasets.MNIST(root='./data',  train=True, download=False, transform=transform)
    testset = tv.datasets.MNIST(root='./data',  train=False, download=False, transform=transform)

    traindata = [i[0].unsqueeze(0) for i in trainset]
    trainlabel = [i[1] for i in trainset]
    testdata = [i[0].unsqueeze(0) for i in testset]
    testlabel = [i[1] for i in testset]

    X = traindata + testdata
    y = trainlabel + testlabel
    X = torch.vstack(X)
    print("X.shape: {}".format(X.shape))        # X.shape: torch.Size([70000, 1, 28, 28])


    # dataset preparation
    graph_constructor =  ConstructUMAPGraph(metric='euclidean', n_neighbors=args.n_neighbors, batch_size=1024, random_state=42)
    epochs_per_sample, head, tail, weight = graph_constructor(X)
    dataset = UMAPDataset(X, epochs_per_sample, head, tail, weight, device=args.device, batch_size=1024)


    # loss function
    if args.DR=="UMAP":
        criterion_DR = UMAPLoss(
            device=args.device, 
            min_dist=args.min_dist, 
            batch_size=args.batch_size, 
            negative_sample_rate=args.negative_sample_rate, 
            edge_weight=None, 
            repulsion_strength=args.repulsion_strength
        )
    elif args.DR=="UMAP":
        pass
    else:
        print("wrong args.DR")
        exit()


    # 0. model initialisation
    model = MMModel(
        MM_I_wPATH=args.MM_I_wPATH, 
        MM_II_wPATH=args.MM_II_wPATH, 
        freeze=(False, True), 
        device=torch.device(args.device)
    )


    # optimisation
    optimizer_DR = torch.optim.Adam(model.parameters(), lr=1e-3)
    # scheduler_I = ...
    optimizer_Human = torch.optim.Adam(model.parameters(), lr=1e-3)
    # scheduler_II = ...


    # 1.1 human model
    model = MMModel(
        MM_I_wPATH="./results/encoder_weights.pt", 
        MM_II_wPATH=None, 
        freeze=(True, False), 
        device=torch.device(args.device)
    )
    # train_Human(model, criterion_Human, optimizer_Human, epochs=args.epochs_Human)

    os.makedirs(os.path.join(result_path, "phase1"))
    print("Go to {} and give feedback.".format(os.path.join(result_path, "phase1")))
    input("Press Enter to continue...")

    # 1.2 DR model
    model, train_losses = train_DR(model, criterion_DR, optimizer_DR, epochs=args.epochs_DR)
    torch.save(model.state_dict(), os.path.join(result_path, 'encoder_weights_1.pt'))
    