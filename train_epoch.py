import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

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

    total_train = train_dataset.batches_per_epoch
    total_test = test_dataset.batches_per_epoch

    # fix Human Model, train DR model
    for param in model.MM_II.parameters():
        param.requires_grad = False

    train_losses = []
    eval_losses = []

    for epoch in range(epochs):

        # train
        train_loss = []
        
        for batch_to, batch_from, batch_index_to, batch_index_from, labels, feedback in tqdm(train_dataset.get_batches(), total=total_train):
        
            y_to = labels[batch_index_to].to(torch.device(device))
            y_from = labels[batch_index_from].to(torch.device(device))

            optimizer.zero_grad()
        
            embedding_to, answers, pref_weights, pred_metrics = model(batch_to, y_to)
            embedding_from, answers, pref_weights, pred_metrics = model(batch_from, y_from)

            loss_DR = criterion(embedding_to, embedding_from)
            loss_HM = F.cross_entropy(input=answers, target=feedback.to(torch.device(device)))

            # metric loss
            weights_metrics = get_weights(pref_weights)
            loss_metrics = torch.mean(pred_metrics * weights_metrics)

            loss = loss_DR + loss_HM + loss_metrics
            # print("train - loss_DR:{}, loss_HM:{}".format(loss_DR.item(), loss_HM.item()))
        
            train_loss.append((loss.item(), loss_DR.item(), loss_HM.item()))
        
            loss.backward()
        
            optimizer.step()

        total_loss = np.mean([i[0] for i in train_loss])
        DR_loss = np.mean([i[1] for i in train_loss])
        HM_loss = np.mean([i[2] for i in train_loss])

        train_losses.append((total_loss, DR_loss, HM_loss))

        print('epoch: {}, loss: {}, '.format(epoch, train_losses))

        # evaluate
        eval_loss = []
        
        for batch_to, batch_from, batch_index_to, batch_index_from, labels, feedback in tqdm(test_dataset.get_batches(), total=total_test):
        
            y_to = labels[batch_index_to].to(torch.device(device))
            y_from = labels[batch_index_from].to(torch.device(device))

            optimizer.zero_grad()
        
            embedding_to, answers, pref_weights, pred_metrics = model(batch_to, y_to)
            embedding_from, answers, pref_weights, pred_metrics = model(batch_from, y_from)
        
            loss_DR = criterion(embedding_to, embedding_from)
            loss_HM = F.cross_entropy(input=answers, target=feedback.to(torch.device(device)))
            loss = loss_DR + loss_HM
            # print("eval - loss_DR:{}, loss_HM:{}".format(loss_DR.item(), loss_HM.item()))
        
            eval_loss.append((loss.item(), loss_DR.item(), loss_HM.item()))
        
            loss.backward()
        
            optimizer.step()

        total_loss = np.mean([i[0] for i in eval_loss])
        DR_loss = np.mean([i[1] for i in eval_loss])
        HM_loss = np.mean([i[2] for i in eval_loss])

        eval_losses.append((total_loss, DR_loss, HM_loss))

        print('DR - epoch: {}, total train_loss: {}, total eval_loss: {}'.format(epoch, train_losses[-1], eval_losses[-1]))

    return model, train_losses, eval_losses


def train_epoch_HM(model, criterion, optimizer, dataloader, epochs=20, device='cuda', scheduler_HM=None, gamma_dab=10):
    """train MM_II and freeze MM_I

    :param model: MM_II
    :param epochs: number of training epochs, defaults to 20
    :return: 
    """

    # fix DR model, train Human Model
    for param in model.MM_I.parameters():
        param.requires_grad = False
    for param in model.MM_II.parameters():
        param.requires_grad = True

    train_losses = []

    for epoch in range(epochs):

        train_loss = []
        
        for X, y, feedback in tqdm(dataloader):
            
            optimizer.zero_grad()
            
            X.requires_grad = True
            pred_z, pred_answers, pref_weights, pred_metrics = model(x=X.to(torch.device(device)), labels=y.to(torch.device(device)))
            
            loss_Qs = criterion(input=pred_answers, target=feedback[0].squeeze().to(torch.device(device)))
            loss_DAB = model.MM_II.scag_module.loss_function().to(torch.device(device))
            loss = loss_Qs + gamma_dab * loss_DAB
            
            # loss_li = calc_multi_loss(answer=pred_answers, feedback=feedback[0].to(torch.device(device)))

            train_loss.append((loss.item(), loss_Qs.item(), loss_DAB.item()))
            
            loss.backward()
        
            optimizer.step()

            # print("mu: {}\nmu grad: {}".format(model.MM_II.mu.view(-1), model.MM_II.mu.grad.view(-1)))
            # print("logvar: {}\nlogvar grad: {}".format(model.MM_II.logvar.view(-1), model.MM_II.logvar.grad.view(-1)))
            # print("user_weights: {}\nuser_weights grad: {}".format(model.MM_II.user_weights.view(-1), model.MM_II.user_weights.grad.view(-1)))

            # for p in model.MM_II.parameters():
            #     if p.data.requires_grad==True:
            #         print(p.name, p.data.shape, p.data.grad.shape, p.data.grad)
            #     else:
            #         print(p.name, p.data.shape)
            # print(model.MM_II.fusion.proj_Q.weight, model.MM_II.fusion.proj_K.weight, model.MM_II.fusion.proj_V.weight)

            # print(model.MM_II.fusion.proj_Q.weight.requires_grad, model.MM_II.fusion.proj_K.weight.requires_grad, model.MM_II.fusion.proj_V.weight.requires_grad)
            # print(model.MM_II.fusion.proj_Q.weight.grad, model.MM_II.fusion.proj_K.weight.grad, model.MM_II.fusion.proj_V.weight.grad)

            # print(model.MM_II.fusion.linear1.weight.requires_grad, model.MM_II.fusion.linear2.weight.requires_grad)
            # print(model.MM_II.fusion.linear1.weight.grad, model.MM_II.fusion.linear2.weight.grad)

        if scheduler_HM!=None:
            scheduler_HM.step()

        total_loss = np.mean([i[0] for i in train_loss])
        Qs_loss = np.mean([i[1] for i in train_loss])
        DAB_loss = np.mean([i[2] for i in train_loss])
        train_losses.append((total_loss, Qs_loss, DAB_loss))

        print('HM - epoch: {}, loss: {}, lr: {}'.format(epoch, train_losses[-1], scheduler_HM.get_lr()[0]))

        if epoch > 0:
            if np.abs(train_losses[-1][0]-train_losses[-2][0])<1e-4:
                break

    return model, train_losses