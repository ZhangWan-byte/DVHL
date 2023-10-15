# https://github.com/SKvtun/ParametricUMAP-Pytorch
import torch
import numpy as np

import torchvision as tv
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.utils import save_image

from models import *

class UMAPDataset:

    def __init__(self, data, labels, epochs_per_sample, head, tail, weight, device='cpu', batch_size=1000):

        """
        create dataset for iteration on graph edges

        """
        self.weigh = weight
        self.batch_size = batch_size
        self.data = data
        self.labels = labels
        self.device = device

        self.edges_to_exp, self.edges_from_exp = (
            np.repeat(head, epochs_per_sample.astype("int")),
            np.repeat(tail, epochs_per_sample.astype("int")),
        )
        self.num_edges = len(self.edges_to_exp)

        # shuffle edges
        shuffle_mask = np.random.permutation(range(len(self.edges_to_exp)))
        self.edges_to_exp = self.edges_to_exp[shuffle_mask]
        self.edges_from_exp = self.edges_from_exp[shuffle_mask]

    def get_batches(self):
        batches_per_epoch = int(self.num_edges / self.batch_size / 5)
        for _ in range(batches_per_epoch):
            rand_index = np.random.randint(0, len(self.edges_to_exp) - 1, size=self.batch_size)
            batch_index_to = self.edges_to_exp[rand_index]
            batch_index_from = self.edges_from_exp[rand_index]

            batch_to = torch.Tensor(self.data[batch_index_to]).to(self.device)
            batch_from = torch.Tensor(self.data[batch_index_from]).to(self.device)
            
            yield (batch_to, batch_from, batch_index_to, batch_index_from, self.labels)


def get_dataset(args, data='MNIST', DR='UMAP'):
    if data=='MNIST':
        # data preprocessing
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        trainTransform  = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Normalize((0.1307,), (0.3081,))])
        trainset = tv.datasets.MNIST(root='./data',  train=True, download=False, transform=transform)
        testset = tv.datasets.MNIST(root='./data',  train=False, download=False, transform=transform)

        X_train = [i[0].unsqueeze(0) for i in trainset]
        X_train = torch.vstack(X_train)
        y_train = torch.tensor([i[1] for i in trainset])
        print("X_train.shape: {}".format(X_train.shape))        # X_train.shape: torch.Size([60000, 1, 28, 28])

        X_test = [i[0].unsqueeze(0) for i in testset]
        X_test = torch.vstack(X_test)
        y_test = torch.tensor([i[1] for i in testset])
        print("X_test.shape: {}".format(X_test.shape))          # X_test.shape: torch.Size([10000, 1, 28, 28])

    if DR=='UMAP':
        # dataset preparation
        train_graph_constructor =  ConstructUMAPGraph(
            metric='euclidean', 
            n_neighbors=args.n_neighbors, 
            batch_size=args.batch_size_DR, 
            random_state=42
        )
        train_epochs_per_sample, train_head, train_tail, train_weight = train_graph_constructor(X_train)
        train_dataset = UMAPDataset(
            X_train, 
            y_train, 
            train_epochs_per_sample, 
            train_head, 
            train_tail, 
            train_weight, 
            device=args.device, 
            batch_size=args.batch_size_DR
        )

        test_graph_constructor =  ConstructUMAPGraph(
            metric='euclidean', 
            n_neighbors=args.n_neighbors, 
            batch_size=args.batch_size_DR, 
            random_state=42
        )
        test_epochs_per_sample, test_head, test_tail, test_weight = test_graph_constructor(X_test)
        test_dataset = UMAPDataset(
            X_test, 
            y_test, 
            test_epochs_per_sample, 
            test_head, 
            test_tail, 
            test_weight, 
            device=args.device, 
            batch_size=args.batch_size_DR
        )

    return train_dataset, test_dataset