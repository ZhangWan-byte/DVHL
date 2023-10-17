import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DRModel import Encoder
from .HumanModel import HumanModel
from .VIModules import VisualImitation


def normalise(z):
    """normalise coordinates to 0~1

    :param z: normalised coordinates
    """
    if type(z) == type(torch.ones(1)):
        z0 = (z[:,0] - torch.min(z[:,0])) / (torch.max(z[:,0]) - torch.min(z[:,0]))
        z1 = (z[:,1] - torch.min(z[:,1])) / (torch.max(z[:,1]) - torch.min(z[:,1]))
        z = torch.hstack([z0.reshape(-1,1), z1.reshape(-1,1)])
    
    if type(z) == type(np.ones(1)):
        z0 = (z[:,0] - np.min(z[:,0])) / (np.max(z[:,0]) - np.min(z[:,0]))
        z1 = (z[:,1] - np.min(z[:,1])) / (np.max(z[:,1]) - np.min(z[:,1]))
        z = np.hstack([z0.reshape(-1,1), z1.reshape(-1,1)])

    return z


class MMModel(nn.Module):
    def __init__(self, MM_I_wPATH, MM_II_wPATH, cnn_layers=[1,1,1,1], VI_size=100, freeze=(False, False), device=torch.device('cuda')):
        super(MMModel, self).__init__()

        # configure MM_I and MM_II
        self.MM_I = Encoder(output_dim=2)
        self.MM_II = HumanModel(cnn_layers=cnn_layers, metric_num=9, device=device)

        if MM_I_wPATH != None or MM_II_wPATH != None:
            self.init_weights(MM_I_wPATH, MM_II_wPATH)

        self.MM_I.to(device)
        self.MM_II.to(device)

        if freeze[0]==True:
            # freeze MM_I - DR model
            self.MM_I.requires_grad_(False)
            self.MM_I.eval()
        if freeze[1]==True:
            # freeze MM_II - Human model
            self.MM_II.requires_grad_(False)
            self.MM_II.eval()

        # configure VisualImitation module
        self.VI = VisualImitation(size=VI_size, device=device)


    def init_weights(self, MM_I_wPATH, MM_II_wPATH):
        if MM_I_wPATH != None:
            print("loading MM_I weights from {}".format(MM_I_wPATH))
            self.MM_I.load_state_dict(torch.load(MM_I_wPATH))
        if MM_II_wPATH != None:
            print("loading MM_II weights from {}".format(MM_II_wPATH))
            self.MM_II.load_state_dict(torch.load(MM_II_wPATH))


    def forward(self, x, labels=None):
               
        z = self.MM_I(x)
        # print("z: ", z.shape)
        
        I_hat = self.VI(z=normalise(z), labels=labels)
        # print("I_hat: ", I_hat.shape)
        
        I_hat = I_hat.permute(2,1,0).unsqueeze(0)
        # print("I_hat: ", I_hat.shape)

        answers = self.MM_II(I_hat=I_hat, z=z, labels=labels, x=x)
        # print("answers: ", answers.shape)
        
        return z, answers