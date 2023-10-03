from .ResNet import *

class HumanModel(nn.Module):
    def __init__(self, cnn_type='resnet18'):
        super(HumanModel, self).__init__()
        
        if cnn_type=="resnet18":
            self.cnn = resnet18()
        elif cnn_type=="resnet34":
            self.cnn = resnet34()
        else:
            print("HumanModel wrong cnn_type!")
            exit()