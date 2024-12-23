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

"""
    MMModel -- old CNN+Pref HumanModel
"""
# class MMModel(nn.Module):
#     def __init__(self, MM_I_wPATH, MM_II_wPATH, cnn_layers=[1,1,1,1], VI_size=100, freeze=(False, False), batch_size=1000, device=torch.device('cuda'), DR='UMAP'):
#         super(MMModel, self).__init__()

#         self.DR = DR

#         self.beta = nn.Parameter(torch.ones(1).to(device))

#         # configure MM_I and MM_II
#         self.MM_I = Encoder(output_dim=2, DR=DR)

#         # self.MM_II = HumanModel(cnn_layers=cnn_layers, metric_num=9, hidden_dim=10, batch_size=batch_size, device=device)
#         self.MM_II = HumanModel(
#             answers_classes=5, 
#             cnn_layers=[1,1,1,1], 
#             metric_num=9, 
#             hidden_dim=10, 
#             out_channels=[10, 16, 24, 32], 
#             batch_size=1000, 
#             device=torch.device('cuda')
#         )
        

#         if MM_I_wPATH != None or MM_II_wPATH != None:
#             self.init_weights(MM_I_wPATH, MM_II_wPATH)

#         self.MM_I.to(device)
#         self.MM_II.to(device)

#         if freeze[0]==True:
#             # freeze MM_I - DR model
#             self.MM_I.requires_grad_(False)
#             self.MM_I.eval()
#         if freeze[1]==True:
#             # freeze MM_II - Human model
#             self.MM_II.requires_grad_(False)
#             self.MM_II.eval()

#         # configure VisualImitation module
#         self.VI = VisualImitation(size=VI_size, device=device)


#     def init_weights(self, MM_I_wPATH, MM_II_wPATH):
#         if MM_I_wPATH != None:
#             print("loading MM_I weights from {}".format(MM_I_wPATH))
#             try:
#                 self.MM_I.load_state_dict(torch.load(MM_I_wPATH))
#             except:
#                 print("alpha and beta not pretrained, both default as 1.0")
#                 MM_I_dict = self.MM_I.state_dict()
#                 pretrained_dict = torch.load(MM_I_wPATH)
#                 pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in MM_I_dict}
#                 MM_I_dict.update(pretrained_dict)
#                 self.MM_I.load_state_dict(MM_I_dict)
#         if MM_II_wPATH != None:
#             print("loading MM_II weights from {}".format(MM_II_wPATH))
#             self.MM_II.load_state_dict(torch.load(MM_II_wPATH))


#     def forward(self, x, labels=None):
               
#         z = self.MM_I(x)
#         # print("z: ", z.shape)
        
#         I_hat = self.VI(z=normalise(z), labels=labels)
#         # print("I_hat: ", I_hat.shape)
        
#         I_hat = I_hat.permute(2,1,0).unsqueeze(0)
#         # print("I_hat: ", I_hat.shape)

#         # answers, pref_weights, pred_metrics = self.MM_II(I_hat=I_hat, z=z, labels=labels, x=x)
#         answers = self.MM_II(I_hat=I_hat, z=z, labels=labels, x=x)
#         # print("answers: ", answers.shape)
        
#         # return z, answers, pref_weights, pred_metrics
#         return z, answers


class MMModel(nn.Module):
    def __init__(self, MM_I_wPATH, MM_II_wPATH, cnn_layers=[1,1,1,1], VI_size=100, freeze=(False, False), batch_size=1000, device=torch.device('cuda'), DR='UMAP'):
        super(MMModel, self).__init__()

        self.DR = DR

        self.beta = nn.Parameter(torch.ones(1).to(device))

        # configure MM_I and MM_II
        self.MM_I = Encoder(output_dim=2, DR=DR)
        self.MM_II = HumanModel(input_size=2, hidden=100, num_classes=10)

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
            try:
                self.MM_I.load_state_dict(torch.load(MM_I_wPATH))
            except:
                print("alpha and beta not pretrained, both default as 1.0")
                MM_I_dict = self.MM_I.state_dict()
                pretrained_dict = torch.load(MM_I_wPATH)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in MM_I_dict}
                MM_I_dict.update(pretrained_dict)
                self.MM_I.load_state_dict(MM_I_dict)
        if MM_II_wPATH != None:
            print("loading MM_II weights from {}".format(MM_II_wPATH))
            self.MM_II.load_state_dict(torch.load(MM_II_wPATH))


    def forward(self, x, labels=None):
               
        z = self.MM_I(x)

        z = normalise(z)

        y = self.MM_II(z)
        
        return z, y