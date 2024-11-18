from copy import deepcopy
from mcts import *
import itertools

import time
import pickle
import argparse
import numpy as np

from env import aug_features

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

all_actions = np.array(
    list(itertools.product([0.8, 1.0, 1.2], [0.8, 1.0, 1.2], [0.8, 1.0, 1.2]))
)

class SearchState():
    def __init__(self, psi_path="surrogate/collaboration_with_ting_zhang/result/rf_0.4.pkl", strategy='avg'):
        self.thetas = [np.array([10, 1.0, 5.0])]
        self.history_actions = []
        self.depth = 0
        self.psi = pickle.load(open(psi_path, 'rb'))

        self.strategy = strategy

    def getPossibleActions(self):
        possibleActions = []
        for i in range(len(all_actions)):
            if np.all(i >= 1 for i in self.thetas[-1] * all_actions[i]):
                possibleActions.append(Action(action_idx=i))
        return possibleActions

    def takeAction(self, action):
        newState = deepcopy(self)
        newState.thetas.append(self.thetas[-1] * all_actions[action.action_idx])
        newState.depth = self.depth + 1
        newState.history_actions.append(action)
        return newState

    def isTerminal(self):
        if self.depth==32 or self.getPossibleActions()==[]:
            return True
        else:
            return False

    def getReward(self):

        if len(self.thetas)>1:

            # 1. average step reward
            if self.strategy == 'avg':
                accumulative_rewards = []
                for i in range(len(self.thetas)):
                    if i==0:
                        continue
                    else:
                        feat = np.array(aug_features(self.thetas[i], self.thetas[i-1], mode=1)).reshape(1,-1)
                        last_r = self.psi.predict_proba(feat)[:, 1]
                        accumulative_rewards.append(last_r)
                r = np.mean(accumulative_rewards)

            # 2. sum reward
            if self.strategy == 'sum':
                accumulative_rewards = []
                for i in range(len(self.thetas)):
                    if i==0:
                        continue
                    else:
                        feat = np.array(aug_features(self.thetas[i], self.thetas[i-1], mode=1)).reshape(1,-1)
                        last_r = self.psi.predict_proba(feat)[:, 1]
                        accumulative_rewards.append(last_r)
                r = np.sum(accumulative_rewards)
        else:
            r = 0

        return r


class Action():
    def __init__(self, action_idx):
        self.action_idx = action_idx

    def __str__(self):
        return str(self.action_idx)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.action_idx == other.action_idx

    def __hash__(self):
        return hash(self.action_idx)


# random action + psi reward
def randomPolicy(state):
    while not state.isTerminal():
        try:
            action = random.choice(state.getPossibleActions())
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.takeAction(action)
    return state.getReward()

# \pi action + psi reward
def pretrainPolicy(state):

    model = my_MLP_forward_embed(
        [128,64,32,27],
        embedding_dim = int(128/2),
        input_dim = 6, 
        device = device
    ).to(device)

    k, m, n = state.thetas[-1][0], state.thetas[-1][1], state.thetas[-1][2]
    feat = torch.tensor([k, m, n, k, m, n]).view(1,-1).float().to(device)
    logits = model(feat, None)

    while not state.isTerminal():
        try:
            while True:
                probs = Categorical(logits=logits)
                action_idx = probs.sample().view(-1).detach().cpu().item()
                if action_idx in [i.action_idx for i in state.getPossibleActions()]:
                    action = Action(action_idx=action_idx)
                    break
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.takeAction(action)
    return state.getReward()


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--simulation', type=str, default="random+avg", help='simulation strategies')
    args = parser.parse_args()

    initialState = SearchState(strategy=args.simulation.split("+")[1])

    print("initialise searcher...")
    if args.simulation.split("+")[0]=='random':
        searcher = mcts(iterationLimit=6000, rolloutPolicy=randomPolicy)
    elif args.simulation.split("+")[0]=='pretrain':
        searcher = mcts(iterationLimit=6000, rolloutPolicy=pretrainPolicy)
    else:
        print("wrong simulation policy!")
        exit()

    print("start searching...")
    t1 = time.time()
    action = searcher.search(initialState=initialState, needDetails=True)
    t2 = time.time()

    print("time used: ", t2-t1)
    for i in range(len(searcher.best_state.history_actions)):
        print("action {}: {}".format(i, searcher.best_state.history_actions[i]))
