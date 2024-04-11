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

cur_time = time.strftime('%m%d%H%M%S', time.localtime())
with open('./exp1_v2/out_{}.txt'.format(cur_time), 'a') as f:
    print("current time: {}".format(cur_time), file=f)

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
with open('./exp1_v2/out_{}.txt'.format(cur_time), 'a') as f:
    print("acquiring names...", file=f)
names = os.listdir("./exp1/data_augmented_v1/")
names = names[:29440] # delete last 5 imgs (46,47,48,49,50)
np.random.shuffle(names)
# names = names[:10000]
train_names = names[:int(len(names)*0.8)]
test_names = names[int(len(names)*0.8):]

with open('./exp1_v2/out_{}.txt'.format(cur_time), 'a') as f:
    print("processing train_dataset...", file=f)
train_dataset = PairPrefDataset(train_names, path="./exp1/data_augmented_v1/", size=256)
z1, z2, y = train_dataset[0]
print("train: ", z1.shape, z2.shape, y)

with open('./exp1_v2/out_{}.txt'.format(cur_time), 'a') as f:
    print("processing test_dataset...", file=f)
test_dataset = PairPrefDataset(test_names, path="./exp1/data_augmented_v1/", size=256)
z1, z2, y = test_dataset[0]
print("test: ", z1.shape, z2.shape, y)

class SiameseNet(nn.Module):
    def __init__(self, hidden, block, num_block, in_channels, out_channels=[10, 16, 32, 64]):
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
    def __init__(self, num_models, hidden, block, num_block, in_channels, out_channels, device):
        super().__init__()
        self.base_models = nn.ModuleList([])
        for _ in range(num_models):
            base = SiameseNet(hidden, block, num_block, in_channels, out_channels).to(device)
            self.base_models.append(base)

    def forward(self, x1, x2):

        indices = np.random.choice(len(self.base_models), round(len(self.base_models)*0.8), replace=False)

        results = []
        for i, base_model in enumerate(self.base_models):
            if i not in indices:
                continue
            out = base_model(x1, x2)
            results.append(out.view(1,-1))
        results = torch.vstack(results).mean(dim=0)
        return results

model = Ensemble(
    num_models=6,
    hidden=64, 
    block=BasicBlock, 
    num_block=[2,2,2,2], #[1,1,1,1],  
    in_channels=1, 
    out_channels=[10, 16, 24, 32], 
    device=torch.device('cuda')
)
print("ensemble params: {}".format(sum([p.numel() for p in model.parameters()])))
with open('./exp1_v2/out_{}.txt'.format(cur_time), 'a') as f:
    print("ensemble params: {}".format(sum([p.numel() for p in model.parameters()])), file=f)

criterion = nn.BCELoss()

epochs = 10
batch_size = 64

optimizer = optim.Adam(model.parameters(), lr=3e-4)
# scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=0.8)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=None)#pref_pair)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=None)#pref_pair)

# Training loop

best_test_loss = 10.0
best_acc = 0.0

train_loss_li = []
test_loss_li = []
test_acc_li = []

for epoch in range(epochs):

    t1 = time.time()

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
            correct += (y_pred.round().view(-1) == y_true.view(-1)).sum().item()

    # test_loss /= len(test_dataloader.dataset)
    test_loss /= batch_size
    accuracy = 100.0 * correct / total

    t2 = time.time()

    print(f"Epoch [{epoch+1}/{epochs}], "
          f"Training Loss: {train_loss:.4f}, "
          f"Test Loss: {test_loss:.4f}, "
          f"Test Accuracy: {accuracy:.2f}%, "
          f"Epoch Time: {t2-t1:.2f}")

    with open('./exp1_v2/out_{}.txt'.format(cur_time), 'a') as f:
        print('\nEpoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.2f}, Time: {:.2f}\n'.format(
            epoch+1, epochs, train_loss, test_loss, accuracy, t2-t1
        ), file=f)  # Python 3.x

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_acc = accuracy
        torch.save(model.state_dict(), "./exp1_v2/model_v1_{}.pt".format(cur_time))

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

torch.save(torch.tensor(test_acc_li), "./exp1_v2/test_acc_{}.pt".format(cur_time))
torch.save(torch.tensor(train_loss_li), "./exp1_v2/train_loss_{}.pt".format(cur_time))
torch.save(torch.tensor(test_loss_li), "./exp1_v2/test_loss_{}.pt".format(cur_time))