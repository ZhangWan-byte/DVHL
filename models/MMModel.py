import torch
import torch.nn as nn
import torch.nn.functional as F

from .MM_I import Encoder
from .HumanModel import HumanModel


class MM(nn.Module):
    def __init__(self, MM_I_wPATH, MM_II_wPATH, freeze=(False, False), device=torch.device('cuda')):
        super(MM, self).__init__()

        # configure MM_I and MM_II
        self.MM_I = Encoder(output_dim=2)
        self.MM_II = HumanModel(cnn_type='resnet18', metric_num=16)

        if MM_I_wPATH != None or MM_II_wPATH != None:
            self.init_weights(MM_I_wPATH, MM_II_wPATH)

        if freeze[0]==True:
            # freeze MM_I
            self.MM_I.requires_grad_(False)
            self.MM_I.eval()
        if freeze[1]==True:
            # freeze MM_II
            self.MM_II.requires_grad_(False)
            self.MM_II.eval()

        # configure VisualImitation module
        self.VI = VisualImitation(size=1000, device=device)


    def init_weights(self, MM_I_wPATH, MM_II_wPATH):
        if MM_I_wPATH != None:
            self.MM_I.load_state_dict(torch.load(MM_I_wPATH))
        if MM_II_wPATH != None:
            self.MM_II.load_state_dict(torch.load(MM_II_wPATH))


    def forward(self, x, labels):
        
        z = self.MM_I(x)

        I_hat = self.VI(z)

        answers = self.MM_II(I_hat=I_hat, z=z, labels=labels, x=x)

        return z, answers