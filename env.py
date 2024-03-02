import gym
from gym import Env
from gym import spaces
import numpy as np
from copy import deepcopy


class DREnv(Env):
    def __init__(self, action_space=12):
        self.rows = 5
        self.cols = 5
        self.start = [0, 0]
        self.goal = [4, 4]
        self.count = 0
        self.current_state = None

        self.action_space = action_space
        self.history_actions = []
        # self.action_space = spaces.Discrete(4)

        # self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([4, 4]))

    def get_initial_state(self, x, label=None, batch_size=1000, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0):
        """generate initial state

        :param x: total data
        :param n_neighbors: kNN, defaults to 10
        :param MN_ratio: mid-pairs, defaults to 0.5
        :param FP_ratio: negatives, defaults to 2.0
        """

        # random sample batch
        idx = torch.randperm(x.shape[0])[:batch_size]
        x = x[idx]
        if label!=None:
            label = label[idx]

        # construct kNN graph -- edge_index and edge_attr
        edge_index = generate_fully_connected_edge_index(num_nodes=x.shape[0])
        A = torch.cdist(x, x, p=2)
        thresholded_A = torch.zeros_like(A, dtype=torch.int)
        for i in range(A.shape[0]):
            _, indices = torch.topk(A[i], n_neighbors+1, largest=False)         # Indices of the k nearest neighbors
            thresholded_A[i, indices] = 1                                       # Set the nearest k entries to 1
        row_indices, col_indices = edge_index
        edge_attr = torch.stack([A[row_indices, col_indices], thresholded_A[row_indices, col_indices]], dim=1)

        # generate state
        state = {
            "x": x, 
            "label": label, 
            "edge_index": edge_index, 
            "edge_attr": edge_attr, 
            
            "history_actions": -1, 
            "effect_history_actions": 0, 

            "n_neighbors": n_neighbors, 
            "MN_ratio": MN_ratio, 
            "FP_ratio": FP_ratio, 

        }

        return state

    def get_new_state(self, action):
        
        alphas_betas = [0.5, 0.8, 1.0, 1.2, 2.0]    # ratio of kNN or mid-pair
        hetero_homo = [0, 1, 2]                     # 0 - random / 1 - hetero / 2 - homo

        # 5 * 5 * 3 = 75 actions
        alpha = alphas_betas[action // 5]
        beta = alphas_betas[action % 5]
        hetero_homo = action % 3


        
    def step(self, action):
        self.count = self.count + 1
        self.history_actions.append(action)

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

        new_state = self.get_new_state(action)
        self.current_state = new_state

        if self.current_state[1] == self.goal[1] and self.current_state[0] == self.goal[0]:
            done = True
            reward = 10.0
        else:
            done = False
            reward = -1
        if self.count > 200:
            done = True

        info = {}
        return self.current_state, reward, done, info

    def render(self):
        pass

    def reset(self):
        self.count = 0
        self.current_state = self.start
        return self.current_state