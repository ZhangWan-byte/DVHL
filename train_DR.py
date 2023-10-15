import torch
import torch.nn as nn
import torch.nn.functional as F

from models import *

def train_DR(model, criterion, optimizer, train_dataset, test_dataset, epochs=20):
    """train MM_I and freeze MM_II

    :param model: MMModel
    :param criterion: loss function
    :param epochs: number of training epochs, defaults to 20
    :return: 
    """

    train_losses = []

    for epoch in range(epochs):

        # train
        train_loss = 0.

        for batch_to, batch_from in tqdm(train_dataset.get_batches()):
        
            optimizer.zero_grad()
        
            embedding_to, _ = model(batch_to)
            embedding_from, _ = model(batch_from)
        
            loss = criterion(embedding_to, embedding_from)
        
            train_loss += loss.item()
        
            loss.backward()
        
            optimizer.step()

        train_losses.append(train_loss.item())

        print('epoch: {}, loss: {}'.format(epoch, train_loss))

        # evaluate
        eval_loss = 0.
        
        for batch_to, batch_from in tqdm(test_dataset.get_batches()):
        
            optimizer.zero_grad()
        
            embedding_to, _ = model(batch_to)
            embedding_from, _ = model(batch_from)
        
            loss = criterion(embedding_to, embedding_from)
        
            train_loss += loss.item()
        
            loss.backward()
        
            optimizer.step()

        train_losses.append(train_loss.item())

        print('epoch: {}, loss: {}'.format(epoch, train_loss))

    return model, train_losses