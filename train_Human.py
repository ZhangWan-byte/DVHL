import torch
import torch.nn as nn
import torch.nn.functional as F

from models import *

def train_Human(model, criterion, optimizer, epochs=20):
    """train MM_II and freeze MM_I

    :param model: MM_II
    :param criterion: loss function
    :param epochs: number of training epochs, defaults to 20
    :return: 
    """

    train_losses = []

    for epoch in range(epochs):

        train_loss = 0.
        
        # for batch_to, batch_from in tqdm(dataset.get_batches()):
        
        #     optimizer.zero_grad()
        
        #     embedding_to = model(batch_to)
        #     embedding_from = model(batch_from)
        
        #     loss = criterion(embedding_to, embedding_from)
        
        #     train_loss += loss.item()
        
        #     loss.backward()
        
        #     optimizer.step()

        train_losses.append(train_loss.item())

        print('epoch: {}, loss: {}'.format(epoch, train_loss))

    return model, train_losses