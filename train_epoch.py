import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

from models import *
from utils import *
from datasets import *


# epoch training for dimensinality reduction
def train_epoch_DR(args, model, criterion, optimizer, scheduler, train_dataset, test_dataset, epochs=20, device='cuda', result_path=None, gamma=0.3):
    """train MM_I and freeze MM_II

    :param model: MMModel
    :param criterion: loss function
    :param epochs: number of training epochs, defaults to 20
    :return: 
    """

    # recording variables
    best_train_loss = (10.0, 10.0)
    best_eval_losses = (10.0, 10.0)
    best_epoch = 0

    train_losses = []
    eval_losses = []

    if args.DR != 'UMAP':
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size_DR, shuffle=True)
        test_loader = DataLoader(train_dataset, batch_size=args.batch_size_DR, shuffle=False)

    for epoch in range(epochs):

        # fix Human Model, train DR model
        for param in model.MM_I.parameters():
            param.requires_grad = True
        for param in model.MM_II.parameters():
            param.requires_grad = False
        model.MM_I.train()
        model.MM_II.eval()

        # train
        train_loss = []
        
        if args.DR == "UMAP":
        
            total_train = train_dataset.batches_per_epoch
            total_test = test_dataset.batches_per_epoch

            for batch_to, batch_from, batch_index_to, batch_index_from, labels, feedback in tqdm(train_dataset.get_batches(), total=total_train):
            
                y_to = labels[batch_index_to].to(torch.device(device))
                y_from = labels[batch_index_from].to(torch.device(device))

                optimizer.zero_grad()
            
                # embedding_to, answers, pref_weights, pred_metrics = model(batch_to, y_to)
                # embedding_from, answers, pref_weights, pred_metrics = model(batch_from, y_from)
                embedding_to, answers = model(batch_to, y_to)
                embedding_from, answers = model(batch_from, y_from)

                loss_DR = criterion(embedding_to, embedding_from)

                # best_labels = torch.ones((answers.shape[0])).int() * 4
                # loss_HM = ord_loss(logits=answers, labels=best_labels.to(torch.device(device)))
                best_labels = torch.ones((answers.shape[0])).long() * 4
                loss_HM = F.cross_entropy(input=answers, target=best_labels.to(torch.device(device)))

                # # metric loss
                # weights_metrics = get_weights(pref_weights)
                # loss_metrics = torch.mean(pred_metrics * weights_metrics)

                # # loss = 0.3*loss_DR + 0.7*loss_HM # + loss_metrics
                # loss = 0.2*loss_DR + 0.4*loss_HM + 0.4*loss_metrics

                # loss = alpha * loss_DR + (1-alpha) * loss_HM

                # PCGrad
                loss = [loss_DR, loss_HM]
                optimizer.pc_backward(loss)
            
                # train_loss.append((loss.item(), loss_DR.item(), loss_HM.item(), loss_metrics.item()))
                train_loss.append((loss_DR.item(), loss_HM.item()))
            
                # loss.backward()
            
                optimizer.step()
        
        elif args.DR == "t-SNE":
            
            for data, labels, feedback in tqdm(train_loader):
            
                data = data.to(torch.device(device))
                labels = labels.to(torch.device(device))

                optimizer.zero_grad()
            
                z, answers = model(data, labels)

                p = calc_p(data, beta=model.MM_I.beta.repeat(data.shape[0]).view(-1,1))         # (batch, batch)
                # q = calc_q(z, alpha=model.MM_I.alpha.repeat(data.shape[0]).view(-1,1))          # (batch, batch)
                q = calc_q(z, alpha=1)

                loss_DR = criterion(p, q)

                best_labels = torch.ones((answers.shape[0])).long() * 4
                loss_HM = F.cross_entropy(input=answers, target=best_labels.to(torch.device(device)))

                loss = gamma * loss_DR + (1-gamma) * loss_HM
            
                train_loss.append((loss_DR.item(), loss_HM.item()))
            
                loss.backward()
            
                optimizer.step()

                # print("alpha grad: {}, beta grad: {}".format(model.MM_I.alpha.grad, model.MM_I.beta.grad))

        else:
            print("wrong args.DR!")
            exit()

        if scheduler!=None:
            scheduler.step()

        DR_loss = np.mean([i[0] for i in train_loss])
        HM_loss = np.mean([i[1] for i in train_loss])
        # Metrics_loss = np.mean([i[2] for i in train_loss])

        train_losses.append((DR_loss, HM_loss))

        # print('DR - train epoch: {}, DR loss: {}, HM_loss: {}, Metrics_loss: {}'.format(
        #     epoch, train_losses[-1][0], train_losses[-1][1], train_losses[-1][2]))
        # print('epoch: {}, DR loss: {}, HM_loss: {}, lr: {}'.format(
        #     epoch, train_losses[-1][0], train_losses[-1][1], scheduler.get_lr()[0]))
        print('epoch: {}, DR loss: {}, HM_loss: {}'.format(
            epoch, train_losses[-1][0], train_losses[-1][1]))

        # evaluate
        with torch.no_grad():

            model.eval()

            eval_loss = []

            if args.DR == 'UMAP':
            
                for batch_to, batch_from, batch_index_to, batch_index_from, labels, feedback in tqdm(test_dataset.get_batches(), total=total_test):
                
                    y_to = labels[batch_index_to].to(torch.device(device))
                    y_from = labels[batch_index_from].to(torch.device(device))

                    optimizer.zero_grad()
                
                    # embedding_to, answers, pref_weights, pred_metrics = model(batch_to, y_to)
                    # embedding_from, answers, pref_weights, pred_metrics = model(batch_from, y_from)
                    embedding_to, answers  = model(batch_to, y_to)
                    embedding_from, answers  = model(batch_from, y_from)
                
                    loss_DR = criterion(embedding_to, embedding_from)

                    # best_labels = torch.ones((answers.shape[0])).int() * 4
                    # loss_HM = ord_loss(logits=answers, labels=best_labels.to(torch.device(device)))
                    best_labels = torch.ones((answers.shape[0])).long() * 4
                    loss_HM = F.cross_entropy(input=answers, target=best_labels.to(torch.device(device)))

                    # # metric loss
                    # weights_metrics = get_weights(pref_weights)
                    # loss_metrics = torch.mean(pred_metrics * weights_metrics)

                    # loss = 0.2*loss_DR + 0.4*loss_HM + 0.4*loss_metrics
                
                    # eval_loss.append((loss.item(), loss_DR.item(), loss_HM.item(), loss_metrics.item()))
                    eval_loss.append((loss_DR.item(), loss_HM.item()))

            elif args.DR == 't-SNE':

                for data, labels, feedback in tqdm(train_loader):
            
                    data = data.to(torch.device(device))
                    labels = labels.to(torch.device(device))

                    optimizer.zero_grad()
                
                    z, answers = model(data, labels)

                    p = calc_p(data, beta=model.MM_I.beta.repeat(data.shape[0]).view(-1,1))
                    # q = calc_q(z, alpha=model.MM_I.alpha.repeat(data.shape[0]).view(-1,1))
                    q = calc_q(z, alpha=1)

                    loss_DR = criterion(p, q)

                    best_labels = torch.ones((answers.shape[0])).long() * 4
                    loss_HM = F.cross_entropy(input=answers, target=best_labels.to(torch.device(device)))

                    loss = gamma * loss_DR + (1-gamma) * loss_HM
                
                    eval_loss.append((loss_DR.item(), loss_HM.item()))

                print("alpha: {}, beta: {}".format(model.MM_I.alpha.item(), model.MM_I.beta.item()))
            
            else:
                print("wrong args.DR!")
                exit()

            DR_loss = np.mean([i[0] for i in eval_loss])
            HM_loss = np.mean([i[1] for i in eval_loss])
            # Metrics_loss = np.mean([i[2] for i in eval_loss])

            eval_losses.append((DR_loss, HM_loss))

            print('DR eval - epoch: {}, DR loss: {}, HM_loss: {}'.format(epoch, eval_losses[-1][0], eval_losses[-1][1]))

            torch.save(model.MM_I.state_dict(), os.path.join(result_path, 'DR_weights_epoch{}.pt'.format(epoch)))

            # if (alpha * eval_losses[-1][0] + (1-alpha) * eval_losses[-1][1]) < sum(best_eval_losses):
            if eval_losses[-1][1] < best_eval_losses[1]:
                torch.save(model.MM_I.state_dict(), os.path.join(result_path, 'DR_weights_best.pt'))
                best_epoch = epoch
                best_eval_losses = eval_losses[-1]
                best_train_loss = train_losses[-1]

        model.train()

    print("best_epoch: {}, best_train_loss: {}, best_test_loss: {}".format(best_epoch, best_train_loss, best_eval_losses))

    return model, train_losses, eval_losses


# epoch training for human model
def train_epoch_HM(model, criterion, optimizer, dataloader, epochs=20, device='cuda', scheduler_HM=None, gamma_dab=10, args=None, result_path=None):
    """train MM_II and freeze MM_I

    :param model: MM_II
    :param epochs: number of training epochs, defaults to 20
    :return: 
    """

    # get feedback test_dataloader
    test_dataloader = get_feedback_loader(batch_size=1)

    # recording variables
    best_train_loss = (10.0, 10.0)
    best_test_loss = (10.0, 10.0)
    best_epoch = 0

    train_losses = []
    test_losses = []

    # training and testing
    for epoch in range(epochs):

        # fix DR model, train Human Model
        for param in model.MM_I.parameters():
            param.requires_grad = False
        for param in model.MM_II.parameters():
            param.requires_grad = True
        
        # fix crowd preference, only learn personal
        model.MM_II.mu.requires_grad = False
        model.MM_II.logvar.requires_grad = False

        # training
        train_loss = []
        for X, y, feedback in tqdm(dataloader):
            
            optimizer.zero_grad()
            
            X.requires_grad = True
            pred_z, pred_answers, pref_weights, pred_metrics = model(x=X.to(torch.device(device)), labels=y.to(torch.device(device)))
            
            # loss_Qs = criterion(input=pred_answers, target=feedback[0].squeeze().to(torch.device(device)))
            loss_Qs = ord_loss(logits=answers, labels=feedbacks.squeeze().cuda())
            loss_DAB = model.MM_II.scag_module.loss_function().to(torch.device(device))
            loss = loss_Qs + gamma_dab * loss_DAB

            train_loss.append((loss.item(), loss_Qs.item(), loss_DAB.item()))
            
            loss.backward()
        
            optimizer.step()

        if scheduler_HM!=None:
            scheduler_HM.step()

        Qs_loss = np.mean([i[0] for i in train_loss])
        DAB_loss = np.mean([i[1] for i in train_loss])
        train_losses.append((Qs_loss, DAB_loss))

        print('HM - epoch: {}, question_loss: {}, DAB_loss: {}, lr: {}'.format(
            epoch, train_losses[-1][0], train_losses[-1][1], scheduler_HM.get_lr()[0]))


        # testing
        with torch.no_grad():

            model.eval()

            test_loss = []
            for z, y, scags, feedbacks in tqdm(test_dataloader):

                z = z.squeeze().float()
                # z.requires_grad = True

                I_hat = model.VI(z=normalise(z).cuda(), labels=y.cuda())
                I_hat = I_hat.permute(2,1,0).unsqueeze(0)
                answers, pref_weights, m = model.MM_II(I_hat=I_hat.cuda(), z=z.cuda(), labels=y.cuda(), x=None)

                # feedbacks.requires_grad = True
                # loss = criterion(input=answers, target=feedbacks.squeeze().cuda())
                loss_Qs = ord_loss(logits=answers, labels=feedbacks.squeeze().cuda())
                loss_DAB = model.MM_II.scag_module.loss_function().to(torch.device(device))
                loss = loss_Qs + gamma_dab * loss_DAB
    
                test_loss.append((loss_Qs.item(), loss_DAB.item()))

            Qs_loss = np.mean([i[0] for i in test_loss])
            DAB_loss = np.mean([i[1] for i in test_loss])
            test_losses.append((Qs_loss, DAB_loss))

            print('HM test - epoch: {}, question_loss: {}, DAB_loss: {}'.format(epoch, test_losses[-1][0], test_losses[-1][1]))
            print("pref_weights: ", pref_weights)
            print("m: ", m)

            if sum(test_losses[-1]) < sum(best_test_loss):
                torch.save(model.MM_II.state_dict(), os.path.join(result_path, 'HM_weights_{}.pt'.format(args.exp_name)))
                best_epoch = epoch
                best_test_loss = test_losses[-1]
                best_train_loss = train_losses[-1]
        
        model.train()

    print("best_epoch: {}, best_train_loss: {}, best_test_loss: {}".format(best_epoch, best_train_loss, best_test_loss))

    return model, train_losses, test_losses