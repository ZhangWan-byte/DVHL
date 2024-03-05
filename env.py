import gym
from gym import Env
from gym import spaces
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F

from utils import *
from myPaCMAP import *

import itertools
import time

class DREnv(Env):
    def __init__(self, x, label, batch_size=1000, action_space=12, history_len=3, save_path=None):
        self.x = x
        self.label = label
        self.batch_size = batch_size
        self.best_reward = 0
        self.best_feedback = 0

        self.count = 0
        self.current_state = None
        self.action_space = action_space

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # history_actions: +1 / 0 / -1
        # effect_history_actions: +1 / 0 / -1
        self.history_len = history_len
        self.history_actions = [0] * history_len
        self.history_rewards = [0] * history_len
        self.effect_history_actions = [0]
        self.history = F.one_hot(
            torch.tensor(self.history_actions + self.effect_history_actions)+1, num_classes=self.action_space
        ).float().to(self.device)           # +1: avoid negative-class in F.one_hot

        self.save_path = save_path

    def obtain_state(self, x, label=None, batch_size=1000, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, initial=False):
        """generate graph -- from params to state (graph)

        :param x: total data, type numpy.float32
        :param n_neighbors: kNN, defaults to 10
        :param MN_ratio: mid-pairs, defaults to 0.5
        :param FP_ratio: negatives, defaults to 2.0
        """
        if initial==True:

            n_neighbors = None
            MN_ratio = 0.5
            FP_ratio = 2.0

        num_nodes = x.shape[0]

        # assign number of neighors / mid-pairs / negatives
        if n_neighbors==None:
            if x.shape[0] <= 10000:
                n_neighbors = 10
            else:
                n_neighbors = int(round(10 + 15 * (np.log10(num_nodes) - 4)))
        n_MN = int(round(n_neighbors * MN_ratio))
        n_FP = int(round(n_neighbors * FP_ratio))

        # generate kNN neighbors, mid-pairs, negatives
        pair_neighbors, pair_MN, pair_FP, tree = generate_pair(
            x, n_neighbors, n_MN, n_FP, distance='euclidean', verbose=False
        )

        # add virtual node
        x = np.vstack([x, np.mean(x, axis=0)])
        
        # edge index
        pair_VN = np.array([(i, num_nodes) for i in range(num_nodes)] + [(num_nodes, i) for i in range(num_nodes)])
        edge_index = np.vstack([pair_neighbors, pair_MN, pair_FP, pair_VN])

        # edge feats
        edge_attr = np.zeros((edge_index.shape[0], 4))
        edge_attr[:pair_neighbors.shape[0], 0] = 1
        edge_attr[pair_neighbors.shape[0]:pair_neighbors.shape[0]+pair_MN.shape[0], 1] = 1
        edge_attr[pair_neighbors.shape[0]+pair_MN.shape[0]:pair_neighbors.shape[0]+pair_MN.shape[0]+pair_VN.shape[0], 2] = 1
        edge_attr[pair_neighbors.shape[0]+pair_MN.shape[0]+pair_VN.shape[0]:, 3] = 1

        edge_index = edge_index.transpose()
        edge_index = edge_index[[1, 0]]         # message: src -> tgt

        # generate state
        state = {
            "x": torch.from_numpy(x).to(self.device), 
            "label": torch.from_numpy(label).to(self.device), 
            "edge_index": torch.from_numpy(edge_index).long().to(self.device), 
            "edge_attr": torch.from_numpy(edge_attr).to(self.device), 

            # "history_actions": self.history_actions, 
            # "effect_history_actions": self.effect_history_actions, 
            "history": self.history.to(self.device), 

            "n_neighbors": n_neighbors, 
            "MN_ratio": MN_ratio, 
            "FP_ratio": FP_ratio, 

            "pair_neighbors": pair_neighbors,
            "pair_MN": pair_MN, 
            "pair_FP": pair_FP
        }

        return state


    def transition(self, action):
        
        # 1. get action
        alpha_values = [0.8, 1.0, 1.2]                # value range of ratio of kNN
        beta_values = [0.8, 1.0, 1.2]                 # value range of ratio of mid-pairs
        gamma_values = [0.8, 1.0, 1.2]                # value range of ratio of negatives
        hetero_homo = [0, 1, 2]                       # 0 - random / 1 - hetero / 2 - homo

        # 3 * 3 * 3 * 3 = 81 actions
        combinations = list(itertools.product(alpha_values, beta_values, gamma_values, hetero_homo))
        alpha, beta, gamma, hetero_homo = combinations[action % len(combinations)]

        n_neighbors = int(alpha * self.current_state["n_neighbors"])
        MN_ratio = beta * self.current_state["MN_ratio"]
        FP_ratio = gamma * self.current_state["FP_ratio"]

        # 2. obtain reward
        reward = self.obtain_reward(self.current_state)
        
        # update best reward
        if reward > self.best_reward:
            self.best_reward = reward

        # 3. obtain history
        # update history actions
        self.history_actions.append(action)
        self.history_actions = self.history_actions[1:]

        # update history rewards
        self.history_rewards.append(reward)
        self.history_rewards = self.history_rewards[1:]

        # add history info to state
        if reward > self.history_rewards[0]:
            self.effect_history_actions = [1]
        else:
            self.effect_history_actions = [-1]

        self.history = F.one_hot(
            torch.tensor(self.history_actions + self.effect_history_actions)+1, num_classes=self.action_space
        ).float().to(self.device)

        # 4. obtain state
        state = self.obtain_state(self.x, self.label, self.batch_size, n_neighbors, MN_ratio, FP_ratio, initial=False)

        return state, reward


    def obtain_reward(self, state):
        with torch.no_grad():

            self.reducer = myPaCMAP(
                n_components=2, 
                n_neighbors=state["n_neighbors"], 
                MN_ratio=state["MN_ratio"], 
                FP_ratio=state["FP_ratio"], 
                pair_neighbors=state["pair_neighbors"], 
                pair_MN=state["pair_MN"], 
                pair_FP=state["pair_FP"]
            )

            name = "iter{}_step{}".format(self.iteration, self.step)
            print("\n\n{} fit-transforming...".format(name))

            t1 = time.time()
            z = self.reducer.fit_transform(self.x)
            t2 = time.time()
            print("time used for fit-transform: {} s".format(t2-t1))

            plt.cla()
            draw_z(
                z=normalise(z), 
                cls=np.ones((z.shape[0], 1)), 
                s=1, 
                save_path=os.path.join(self.save_path, name), 
                display=False, 
                title=name, 
                palette=None
            )
            features = np.zeros((5, 1))
            print("Please refer to image {}.".format(os.path.join(self.save_path, name)))
            features[0] = int(input("\n1. How many clusters?\n\tcount number of clusters\n"))
            features[1] = int(input("\n2. Rate the shape of these clusters from 1 to 4\n\t'round -> oval -> spindle-shaped -> linear'.\n"))
            features[2] = int(input("\n3. How many 'connections' between clusters?\n\tfrom one cluster, you know what's 'next' cluster\n"))
            features[3] = int(input("\n4. Can you observe an obvious trend or ordinal relations between clusters? Scores from 1 to 5.\n\t1 - totally not\n\t2 - not obvious\n\t3 - partly trend\n\t4 - partial trend, need imagination\n\t5 - explicit trend\n"))
            features[4] = int(input("\n5. Do you like this visualisation? Scores from 1 to 5.\n"))
            np.save(os.path.join(self.save_path, "{}.npy".format(name)), features.reshape(1, -1))
            
            feedback = features[-1]

            # r1: compared to last reward
            if feedback > self.history_rewards[-1]:
                r1 = 1
            else:
                r1 = 0

            # r2: compared to best reward
            r2 = feedback - self.best_feedback
            if r2>0:
                self.best_feedback = feedback

            return r1 + r2
        
    def next_step(self, action, iteration, step):
        self.count = self.count + 1
        self.iteration = iteration
        self.step = step

        # new_state = deepcopy(self.current_state)

        # if action == 0:  # up
        #     new_state[0] = max(new_state[0] - 1, 0)
        # elif action == 1:  # down
        #     new_state[0] = min(new_state[0] + 1, self.cols - 1)
        # elif action == 2:  # left
        #     new_state[1] = max(new_state[1] - 1, 0)
        # elif action == 3:  # right
        #     new_state[1] = min(new_state[1] + 1, self.rows - 1)
        # else:
        #     raise Exception("Invalid action")

        new_state, reward = self.transition(action)
        self.current_state = new_state

        # if self.current_state[1] == self.goal[1] and self.current_state[0] == self.goal[0]:
        #     done = True
        #     reward = 10.0
        # else:
        #     done = False
        #     reward = -1
        # if self.count > 200:
        #     done = True

        info = {}
        terminations = 0 # accident or illegal situation
        done = 0 if self.count<10 else 1

        return self.current_state, reward, terminations, done, info

    def render(self):
        pass

    def reset(self):
        self.count = 0
        self.iteration = 1
        self.step = 0

        self.current_state = self.obtain_state(self.x, self.label, self.batch_size, initial=True) # self.start
        return self.current_state