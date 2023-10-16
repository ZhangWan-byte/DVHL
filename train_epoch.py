import torch
import torch.nn as nn
import torch.nn.functional as F

from models import *


def calc_multi_loss(answer, feedback):
    """calculate question-answer loss

    :param feedback: predicted answer
    :param feedback: feedback
    """

    loss_li = []

    for i in range(len(feedback)):
        loss_i = F.cross_entropy(input=answer[i].view(1,-1), target=feedback[i].view(1,-1))
        loss_li.append(loss_i.item())

    return loss_li


def train_epoch_DR(model, criterion, optimizer, train_dataset, test_dataset, epochs=20, device='cuda'):
    """train MM_I and freeze MM_II

    :param model: MMModel
    :param criterion: loss function
    :param epochs: number of training epochs, defaults to 20
    :return: 
    """

    total_train = int(np.ceil(len(train_dataset.data)/train_dataset.batch_size))
    total_test = int(np.ceil(len(test_dataset.data)/test_dataset.batch_size))

    # fix Human Model, train DR model
    for param in model.MM_II.parameters():
        param.requires_grad = False

    train_losses = []
    eval_losses = []

    for epoch in range(epochs):

        # train
        train_loss = []
        
        for batch_to, batch_from, batch_index_to, batch_index_from, labels in tqdm(train_dataset.get_batches(), total=total_train):
        
            y_to = labels[batch_index_to].to(torch.device(device))
            y_from = labels[batch_index_from].to(torch.device(device))

            optimizer.zero_grad()
        
            embedding_to, answers = model(batch_to, y_to)
            embedding_from, answers = model(batch_from, y_from)

            loss = criterion(embedding_to, embedding_from)
        
            train_loss.append(loss.item())
        
            loss.backward()
        
            optimizer.step()

        train_losses.append(np.mean(train_loss))

        print('epoch: {}, loss: {}'.format(epoch, train_loss))

        # evaluate
        eval_loss = []
        
        for batch_to, batch_from, batch_index_to, batch_index_from, labels in tqdm(test_dataset.get_batches(), total=total_test):
        
            y_to = labels[batch_index_to].to(torch.device(device))
            y_from = labels[batch_index_from].to(torch.device(device))

            optimizer.zero_grad()
        
            embedding_to, answers = model(batch_to, y_to)
            embedding_from, answers = model(batch_from, y_from)
        
            loss = criterion(embedding_to, embedding_from)
        
            eval_loss.append(loss.item())
        
            loss.backward()
        
            optimizer.step()

        eval_losses.append(np.mean(eval_loss))

        print('DR - epoch: {}, train_loss: {}, eval_loss'.format(epoch, train_losses[-1], eval_losses[-1]))

    return model, train_losses, eval_losses


def train_epoch_HM(model, criterion, optimizer, dataloader, epochs=20, device='cuda'):
    """train MM_II and freeze MM_I

    :param model: MM_II
    :param epochs: number of training epochs, defaults to 20
    :return: 
    """

    # fix DR model, train Human Model
    for param in model.MM_I.parameters():
        param.requires_grad = False

    train_losses = []

    for epoch in range(epochs):

        train_loss = []
        
        for X, y, feedback in tqdm(dataloader):
            # print("feedback: ", feedback.shape)
            optimizer.zero_grad()
        
            pred_z, pred_answers = model(x=X.to(torch.device(device)), labels=y.to(torch.device(device)))
            # print("1 pred_z: {}\npred_answers: {}".format(pred_z.shape, pred_answers))
            loss = criterion(input=pred_answers, target=feedback[0].squeeze().to(torch.device(device)))
            # print("2 pred_z: {}\npred_answers: {}".format(pred_z.shape, pred_answers))

            # loss_li = calc_multi_loss(answer=pred_answers, feedback=feedback[0].to(torch.device(device)))

            train_loss.append(loss.item())
            # print("loss_li, ", loss_li)
            
            loss.backward()
        
            optimizer.step()

        train_losses.append(np.mean(train_loss))

        print('HM - epoch: {}, loss: {}'.format(epoch, train_losses[-1]))

    return model, train_losses