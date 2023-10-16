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

from train_epoch import train_epoch_HM


if __name__=='__main__':


    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='HM', help="train DR or HM")
    parser.add_argument('--MM_I_wPATH', type=str, default="./results/DR_weights.pt", help='weights to initialise DR model')
    parser.add_argument('--MM_II_wPATH', type=str, default=None, help='weights to initialise human model')
    
    # training params
    parser.add_argument('--device', type=str, default='cuda', help='device cpu or cuda')

    parser.add_argument('--batch_size_HM', type=int, default=1, help='batch size - phase Human')
    parser.add_argument('--epochs_HM', type=int, default=100, help='training epochs - phase Human')

    # data
    parser.add_argument('--feedback_path', type=str, default=None, help='path of human feedback')

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


    # human model
    model = MMModel(
        MM_I_wPATH=args.MM_I_wPATH, 
        MM_II_wPATH=None, 
        cnn_layers=[1,1,1,1], 
        VI_size=100, 
        freeze=(True, False), 
        device=torch.device(args.device)
    )

    # get dataloader
    train_loader, test_loader = get_dataset(data='MNIST', DR='UMAP', args=args)

    # loss function
    criterion_HM = nn.CrossEntropyLoss()

    # optimisation
    optimizer_HM = torch.optim.Adam(model.MM_II.parameters(), lr=1e-3)
    # # scheduler_II = ...

    model, train_losses = train_epoch_HM(
        model, 
        criterion_HM, 
        optimizer_HM, 
        train_loader, 
        epochs=args.epochs_HM
    )

    torch.save(model.MM_II.state_dict(), os.path.join(result_path, 'HM_weights_{}.pt'.format(exp_name)))
    