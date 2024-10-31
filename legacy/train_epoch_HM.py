import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

from models import *
from utils import *
from datasets import *


# epoch training for human model
def train_epoch_HM(model, criterion, optimizer, train_loader, test_loader, epochs=20, device='cuda', scheduler_HM=None, gamma_dab=10, args=None, result_path=None):
    """train MM_II and freeze MM_I

    :param model: MM_II
    :param epochs: number of training epochs, defaults to 20
    :return: 
    """

    # # get feedback test_dataloader
    # test_dataloader = get_feedback_loader(batch_size=1)

    # recording variables
    # best_train_loss = (10.0, 10.0)
    # best_test_loss = (10.0, 10.0)
    best_train_loss = 10.0
    best_test_loss = 10.0
    best_epoch = 0

    train_losses = []
    test_losses = []

    # training and testing
    for epoch in range(epochs):

        # # fix DR model, train Human Model
        # for param in model.MM_I.parameters():
        #     param.requires_grad = False
        # for param in model.MM_II.parameters():
        #     param.requires_grad = True
        
        # # fix crowd preference, only learn personal
        # model.MM_II.mu.requires_grad = False
        # model.MM_II.logvar.requires_grad = False

        # training
        train_loss = []
        for X, y, feedback in tqdm(train_loader):

            optimizer.zero_grad()
            
            X.requires_grad = True
            # pred_z, pred_answers, pref_weights, pred_metrics = model(x=X.to(torch.device(device)), labels=y.to(torch.device(device)))
            pred_y = model(X.to(torch.device(device)))
            
            # # loss_Qs = criterion(input=pred_answers, target=feedback[0].squeeze().to(torch.device(device)))
            # loss_Qs = ord_loss(logits=answers, labels=feedbacks.squeeze().cuda())
            # loss_DAB = model.MM_II.scag_module.loss_function().to(torch.device(device))
            # loss = loss_Qs + gamma_dab * loss_DAB

            # train_loss.append((loss.item(), loss_Qs.item(), loss_DAB.item()))

            loss = criterion(pred_y, y.to(torch.device(device))) + 1e-3 * sum(p.pow(2).sum() for p in model.parameters())
            # 1e-3 * sum(p.pow(2).sum() for p in model.MM_II.parameters())
            
            loss.backward()
        
            optimizer.step()

            train_loss.append(loss.item())

        if scheduler_HM!=None:
            scheduler_HM.step()

        # Qs_loss = np.mean([i[0] for i in train_loss])
        # DAB_loss = np.mean([i[1] for i in train_loss])
        # train_losses.append((Qs_loss, DAB_loss))
        train_losses.append(np.mean(train_loss))

        # print('HM - epoch: {}, question_loss: {}, DAB_loss: {}, lr: {}'.format(
        #     epoch, train_losses[-1][0], train_losses[-1][1], scheduler_HM.get_lr()[0]))

        # testing
        with torch.no_grad():

            model.eval()

            test_loss = []
            y_preds = []
            y_trues = []
            # for z, y, scags, feedbacks in tqdm(test_dataloader):
            for X, y, feedback in tqdm(test_loader):

                # z = z.squeeze().float()
                # # z.requires_grad = True

                # I_hat = model.VI(z=normalise(z).cuda(), labels=y.cuda())
                # I_hat = I_hat.permute(2,1,0).unsqueeze(0)
                # answers, pref_weights, m = model.MM_II(I_hat=I_hat.cuda(), z=z.cuda(), labels=y.cuda(), x=None)

                # # feedbacks.requires_grad = True
                # # loss = criterion(input=answers, target=feedbacks.squeeze().cuda())
                # loss_Qs = ord_loss(logits=answers, labels=feedbacks.squeeze().cuda())
                # loss_DAB = model.MM_II.scag_module.loss_function().to(torch.device(device))
                # loss = loss_Qs + gamma_dab * loss_DAB
    
                # test_loss.append((loss_Qs.item(), loss_DAB.item()))

                # pred_z, pred_y = model(X.to(torch.device(device)), labels=y.to(torch.device(device)))
                pred_y = model(X.to(torch.device(device)))

                y_preds.append(torch.argmax(pred_y.detach().cpu(), dim=1).numpy())
                y_trues.append(torch.argmax(y.detach().cpu(), dim=1).numpy())

                loss = criterion(pred_y, y.to(torch.device(device)))

                test_loss.append(loss.item())

            # Qs_loss = np.mean([i[0] for i in test_loss])
            # DAB_loss = np.mean([i[1] for i in test_loss])
            # test_losses.append((Qs_loss, DAB_loss))
            test_losses.append(np.mean(test_loss))

            # print('HM test - epoch: {}, question_loss: {}, DAB_loss: {}'.format(epoch, test_losses[-1][0], test_losses[-1][1]))
            # print("pref_weights: ", pref_weights)
            # print("m: ", m)
            lr = scheduler_HM.get_lr()[0] if scheduler_HM!=None else optimizer.param_groups[0]['lr']
            print("HM - epoch: {}, train_loss: {}, test_loss: {}, lr: {}".format(
                epoch, train_losses[-1], test_losses[-1], lr))

            # if sum(test_losses[-1]) < sum(best_test_loss):
            if test_losses[-1] < best_test_loss:
                # torch.save(model.MM_II.state_dict(), os.path.join(result_path, 'HM_weights_{}.pt'.format(args.exp_name)))
                torch.save(model.state_dict(), os.path.join(result_path, 'HM_weights_{}.pt'.format(args.exp_name)))
                best_epoch = epoch
                best_test_loss = test_losses[-1]
                best_train_loss = train_losses[-1]
        
        model.train()

    print("best_epoch: {}, best_train_loss: {}, best_test_loss: {}".format(
        best_epoch, best_train_loss, best_test_loss))

    return model, train_losses, test_losses