import os
import time
import copy
import pickle
import numpy as np
import pandas as pd
import seaborn as sn
from tqdm import tqdm
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import sklearn
from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap, MDS, SpectralEmbedding
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Function

from models import *
from utils import *
from datasets import *


def get_Ihat(Z, size=1000):
    mat = np.zeros((size,size))
    for i in range(len(Z)):
        xx = min(int(np.floor(Z[i,0]*size)), size-1)
        yy = min(int(np.floor(Z[i,1]*size)), size-1)
        if Z.shape[1]==3:
            mat[xx,yy] = Z[i, 2]
        else:
            mat[xx,yy] = 1
    return mat

def flip(z):
    x = z[:, 0]
    y = z[:, 1]
    if np.random.rand()<0.5:
        x = 1 - x
    else:
        y = 1 - y
    
    try:
        return torch.hstack([x.view(-1,1), y.view(-1,1)])
    except:
        return np.hstack([x.reshape(-1,1), y.reshape(-1,1)])


class PairPrefDataset(Dataset):
    def __init__(self, names, path="./exp1/data_augmented_v1/", size=256):
        
        self.names = names
        self.path = path
        # self.all_z1 = []
        # self.all_z2 = []
        # self.all_y = []
        # for i in tqdm(range(len(names))):
        #     z1, z2, y = torch.load(os.path.join(path, self.names[i]))

        #     z1 = torch.from_numpy(z1).unsqueeze(0).float()
        #     z2 = torch.from_numpy(z2).unsqueeze(0).float()

        #     self.all_z1.append(z1)
        #     self.all_z2.append(z2)
        #     self.all_y.append(y)

        # self.all_z1 = torch.vstack(self.all_z1).float()
        # self.all_z2 = torch.vstack(self.all_z2).float()
        # self.y = torch.tensor(self.all_y).float()

        print("total samples: ", len(self.names))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        # z1 = self.all_z1[idx]
        # z2 = self.all_z2[idx]
        z1, z2, y = torch.load(os.path.join(self.path, self.names[idx]))
        z1 = torch.from_numpy(z1).unsqueeze(0).float()
        z2 = torch.from_numpy(z2).unsqueeze(0).float()
        y = torch.tensor([y]).float()

        return z1, z2, y

print("acquiring names...")
names = os.listdir("./exp1/data_augmented_v1/")
np.random.shuffle(names)
train_names = names[:int(len(names)*0.8)]
test_names = names[int(len(names)*0.8):]

print("processing train_dataset...")
train_dataset = PairPrefDataset(train_names, path="./exp1/data_augmented_v1/", size=256)
z1, z2, y = train_dataset[0]
print("train: ", z1.shape, z2.shape, y)

print("processing test_dataset...")
test_dataset = PairPrefDataset(test_names, path="./exp1/data_augmented_v1/", size=256)
z1, z2, y = test_dataset[0]
print("test: ", z1.shape, z2.shape, y)

class SiameseNet(nn.Module):
    def __init__(self, hidden, block, num_block, in_channels, out_channels=[10, 16, 32, 64], num_classes=16):
        super().__init__()
        self.cnn = ResNet(
            block=BasicBlock, 
            num_block=[1,1,1,1], 
            num_classes=hidden, 
            in_channels=1, 
            out_channels=[10, 16, 24, 32])

        self.hidden = hidden

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden*2, self.hidden), 
            nn.ReLU(), 
            nn.Dropout(p=0.5), 
            nn.Linear(self.hidden, 1)
        )

    def forward(self, x1, x2):
        emb1 = self.cnn(x1)
        emb2 = self.cnn(x2)

        x = torch.hstack([emb1, emb2])
        x = self.mlp(x)
        x = F.sigmoid(x)

        return x

class Ensemble(nn.Module):
    def __init__(self, num_models, hidden, block, num_block, in_channels, out_channels, num_classes, device):
        super().__init__()
        self.base_models = nn.ModuleList([])
        for _ in range(num_models):
            base = SiameseNet(hidden, block, num_block, in_channels, out_channels, num_classes).to(device)
            self.base_models.append(base)

    def forward(self, x1, x2):
        results = []
        for base_model in self.base_models:
            out = base_model(x1, x2)
            results.append(out.view(1,-1))
        results = torch.vstack(results).mean(dim=0)
        return results

model = Ensemble(
    num_models=6,
    hidden=64, 
    block=BasicBlock, 
    num_block=[1,1,1,1], 
    num_classes=5, 
    in_channels=1, 
    out_channels=[10, 16, 24, 32], 
    device=torch.device('cuda')
)
print("ensemble params: {}".format(sum([p.numel() for p in model.parameters()])))

criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=3e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)

# Training loop
epochs = 100
batch_size = 32

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=None)#pref_pair)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=None)#pref_pair)

ref_cls = torch.tensor([0, 1, 2, 3, 4]).cuda()

best_test_loss = 10.0
best_acc = 0.0

train_loss_li = []
test_loss_li = []
test_acc_li = []

for epoch in range(epochs):

    # Training
    model.train()
    train_loss = 0.0
    for x1, x2, y in train_dataloader:
        optimizer.zero_grad()

        x1 = x1.cuda()
        x2 = x2.cuda()
        y = y.cuda()

        outputs = model(x1, x2)
        loss = criterion(outputs.view(-1), y.view(-1))
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    # train_loss /= len(train_dataloader.dataset)
    train_loss /= batch_size

    # Testing
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x1, x2, y in test_dataloader:
            x1 = x1.cuda()
            x2 = x2.cuda()
            y = y.cuda()
    
            outputs = model(x1, x2)
            loss = criterion(outputs.view(-1), y.view(-1))
            test_loss += loss.item()

            y_pred = outputs.detach().cpu()
            y_true = y.detach().cpu()

            total += y.size(0)
            correct += (y_pred.round() == y_true).sum().item()

    # test_loss /= len(test_dataloader.dataset)
    test_loss /= 100
    accuracy = 100.0 * correct / total

    print(f"Epoch [{epoch+1}/{epochs}], "
          f"Training Loss: {train_loss:.4f}, "
          f"Test Loss: {test_loss:.4f}, "
          f"Test Accuracy: {accuracy:.2f}%")

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_acc = accuracy
        torch.save(model.state_dict(), "./exp1/model_v1.pt")

    train_loss_li.append(train_loss)
    test_loss_li.append(test_loss)
    test_acc_li.append(accuracy)

    scheduler.step()


# plt.figure(figsize=(16,6))

# plt.subplot(121)
# plt.plot(test_acc_li)
# plt.legend(["test acc"])

# plt.subplot(122)
# plt.plot(train_loss_li)
# plt.plot(test_loss_li)
# plt.legend(["train loss", "test loss"])
# plt.show()

torch.save(torch.tensor(test_acc_li), "./exp1/test_acc.pt")
torch.save(torch.tensor(train_loss_li), "./exp1/train_loss.pt")
torch.save(torch.tensor(test_loss_li), "./exp1/test_loss.pt")