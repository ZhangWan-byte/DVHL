import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

from models import *
from utils import *


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
        with torch.no_grad():

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

            print('DR - epoch: {}\ntotal train_loss: {}\ntotal eval_loss: {}'.format(epoch, train_losses[-1], eval_losses[-1]))

    return model, train_losses, eval_losses


def train_epoch_HM(model, criterion, optimizer, dataloader, epochs=20, device='cuda', scheduler_HM=None, gamma_dab=10, args=None):
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

    # get feedback test_dataloader
    test_dataloader = get_feedback_loader()

    # recording variables
    best_train_loss = 10.0
    best_test_loss = 10.0
    best_epoch = 0

    train_losses = []
    test_losses = []

    for epoch in range(epochs):

        # training
        train_loss = []
        for X, y, feedback in tqdm(dataloader):
            
            optimizer.zero_grad()
            
            X.requires_grad = True
            pred_z, pred_answers, pref_weights, pred_metrics = model(x=X.to(torch.device(device)), labels=y.to(torch.device(device)))
            
            loss_Qs = criterion(input=pred_answers, target=feedback[0].squeeze().to(torch.device(device)))
            loss_DAB = model.MM_II.scag_module.loss_function().to(torch.device(device))
            loss = loss_Qs + gamma_dab * loss_DAB

            train_loss.append((loss.item(), loss_Qs.item(), loss_DAB.item()))
            
            loss.backward()
        
            optimizer.step()

        if scheduler_HM!=None:
            scheduler_HM.step()

        total_loss = np.mean([i[0] for i in train_loss])
        Qs_loss = np.mean([i[1] for i in train_loss])
        DAB_loss = np.mean([i[2] for i in train_loss])
        train_losses.append((total_loss, Qs_loss, DAB_loss))

        print('HM - epoch: {}, question_loss: {}, DAB_loss: {}, lr: {}'.format(
            epoch, train_losses[-1][1], train_losses[-1][2], scheduler_HM.get_lr()[0]))


        # testing
        test_loss = []
        for z, y, scags, feedbacks in tqdm(test_dataloader):

            z = z.squeeze().float()
            # z.requires_grad = True

            I_hat = model.VI(z=normalise(z).cuda(), labels=y.cuda())
            I_hat = I_hat.permute(2,1,0).unsqueeze(0)
            answers, pref_weights, m = model.MM_II(I_hat=I_hat.cuda(), z=z.cuda(), labels=y.cuda(), x=None)

            # feedbacks.requires_grad = True
            loss = criterion(input=answers, target=feedbacks.squeeze().cuda())
            
            test_loss.append(loss.item())

        test_losses.append(np.mean(test_loss))

        print('HM - test: {}, lr: {}'.format(epoch, test_losses[-1]))
        print("pref_weights: ", pref_weights)
        print("m: ", m)

        if test_losses[-1] < best_test_loss:
            torch.save(model.MM_II.state_dict(), os.path.join(result_path, 'HM_weights_{}.pt'.format(args.exp_name)))
            best_epoch = epoch
            best_test_loss = test_losses[-1]
            best_train_loss = train_losses[-1]

    print(best_epoch, best_train_loss, best_test_loss)

    return model, train_losses, test_losses