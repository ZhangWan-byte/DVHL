import gym
from gym import Env
from gym import spaces
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F

from utils import *
from models import *
from myPaCMAP import *

import itertools
import time

# from memory_profiler import profile

# human surrogate
class SiameseNet(nn.Module):
    def __init__(self, hidden, block, num_block, in_channels, out_channels=[10, 16, 32, 64], num_classes=16):
        super().__init__()
        self.cnn = ResNet(
            block=BasicBlock, 
            num_block=[1,1,1,1], 
            num_classes=hidden, 
            in_channels=1, 
            out_channels=[10, 16, 24, 32])

        self.hidden = hidden

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden*2, self.hidden), 
            nn.ReLU(), 
            nn.Dropout(p=0.5), 
            nn.Linear(self.hidden, 1)
        )

    def forward(self, x1, x2):
        emb1 = self.cnn(x1)
        emb2 = self.cnn(x2)

        x = torch.hstack([emb1, emb2])
        x = self.mlp(x)
        x = F.sigmoid(x)

        return x


class DREnv(Env):
    def __init__(self, x, label, model_path="./exp1/model_CosAnneal1.pt", action_space=27, history_len=7, save_path=None, num_steps=32, num_partition=20, size=256, run_name=None):
        self.x = x
        self.label = label
        self.best_reward = 0
        self.best_feedback = 0
        self.name = None
        self.best_name = None
        self.size = size

        self.run_name = run_name

        self.count = 0
        self.current_state = None
        self.action_space = action_space

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_partition = num_partition

        # human surrogate
        self.model = SiameseNet(
            hidden=64, 
            block=BasicBlock, 
            num_block=[1,1,1,1], 
            num_classes=5, 
            in_channels=1, 
            out_channels=[10, 16, 24, 32]
        ).cuda()
        self.model.load_state_dict(torch.load(model_path))
        print("human surrogate params: ", sum([p.numel() for p in self.model.parameters()]))
        with open("./runs/{}/println.txt".format(self.run_name), 'a') as f:
            print("human surrogate params: ", sum([p.numel() for p in self.model.parameters()]), file=f)
        self.model.train()

        # actions
        self.alpha_values = [0.8, 1.0, 1.2]                # value range of ratio of kNN
        self.beta_values = [0.8, 1.0, 1.2]                 # value range of ratio of mid-pairs
        self.gamma_values = [0.8, 1.0, 1.2]                # value range of ratio of negatives
        # self.hetero_homo = [0, 1, 2]                       # 0 - random / 1 - hetero / 2 - homo
        # 3 * 3 * 3 = 27 actions (0,1,...,26)
        self.combinations = np.array(
            list(itertools.product(self.alpha_values, self.beta_values, self.gamma_values))
        )

        # best & last vis
        self.best_z = None
        self.best_z0 = None
        self.last_z = None

        # history: actions / effect
        self.history_len = history_len
        self.history_actions = torch.zeros((history_len, self.num_partition))
        self.history_rewards = torch.tensor([])
        self.effect_history_actions = torch.zeros(1)

        # like a sentence, abcdefg0 / fdsjkhy1, each character is one_hot
        self.history = [self.history_actions, self.effect_history_actions]
        self.history_feedbacks = []

        self.save_path = save_path

        self.num_steps = num_steps

    def obtain_state(self, x, label=None, n_neighbors=None, MN_ratio=None, FP_ratio=None, initial=False):
        """generate graph -- from params to state (graph)

        :param x: total data, type numpy.float32
        :param n_neighbors: kNN, defaults to 10
        :param MN_ratio: mid-pairs, defaults to 0.5
        :param FP_ratio: negatives, defaults to 2.0
        """
        if initial==True:

            n_neighbors = np.round(np.ones(x.shape[0]) * 15).astype(np.int32) #None
            MN_ratio = np.ones(x.shape[0]) * 2.0 # 0.5
            FP_ratio = np.ones(x.shape[0]) * 20.0 # 2.0

        num_nodes = x.shape[0]

        # assign number of neighors / mid-pairs / negatives
        if n_neighbors is None:
            if x.shape[0] <= 10000:
                n_neighbors = np.round(np.ones(x.shape[0]) * 10).astype(np.int32)
            else:
                n_neighbors = np.round(np.ones(x.shape[0]) * int(round(10 + 15 * (np.log10(num_nodes) - 4)))).astype(np.int32)
        n_MN = np.round(n_neighbors * MN_ratio).astype(np.int32)
        n_FP = np.round(n_neighbors * FP_ratio).astype(np.int32)

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
        edge_attr[:pair_neighbors.shape[0], 0] = 1                                                      # knn
        edge_attr[pair_neighbors.shape[0]:pair_neighbors.shape[0]+pair_MN.shape[0], 1] = 1              # mid-pair
        edge_attr[pair_neighbors.shape[0]+pair_MN.shape[0]:pair_neighbors.shape[0]+pair_MN.shape[0]+pair_FP.shape[0], 2] = 1    # FP
        edge_attr[pair_neighbors.shape[0]+pair_MN.shape[0]+pair_FP.shape[0]:, 3] = 1                    # VN

        edge_index = edge_index.transpose()
        edge_index = edge_index[[1, 0]]         # message: src -> tgt

        # generate state
        state = {
            "x": torch.from_numpy(x).to(self.device), 
            "label": torch.from_numpy(label).to(self.device), 
            "edge_index": torch.from_numpy(edge_index).long().to(self.device), 
            "edge_attr": torch.from_numpy(edge_attr).to(self.device), 
 
            "history": [i.to(self.device) for i in self.history], 

            "n_neighbors": n_neighbors, 
            "MN_ratio": MN_ratio, 
            "FP_ratio": FP_ratio, 

            "pair_neighbors": pair_neighbors,
            "pair_MN": pair_MN, 
            "pair_FP": pair_FP
        }

        return state

    def heuristic(self, out_mean, out_var, mode=2, t1=0.02, t2=0.5):

        if mode==0:
        # constructively conservative and discrete and sparse
            if out_var < t1:
                r = out_mean.round()
            else:
                if out_mean > t2:
                    r = max((out_mean - np.sqrt(out_var)*3).round().item(), 0)
                else:
                    r = min((out_mean + np.sqrt(out_var)*3).round().item(), 1)

        elif mode==1:
        # constructively conservative and continuous and dense
            if out_var < t1:
                r = out_mean
            else:
                if out_mean > t2:
                    r = max((out_mean - np.sqrt(out_var)).item(), 0)
                else:
                    r = min((out_mean + np.sqrt(out_var)).item(), 1)

        elif mode==2:
        # conservative and continuous and dense
            r = max((out_mean - np.sqrt(out_var)).item(), 0)

        elif mode==3:
        # conservative and discrete and sparse
            r = max((out_mean - np.sqrt(out_var)*3).round().item(), 0)

        else:
            print("not implemented heuristic!")
            exit()

        return r

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
            with open("./runs/{}/println.txt".format(self.run_name), 'a') as f:
                print("\n\n{} fit-transforming...".format(name), file=f)

            t1 = time.time()
            z0 = self.reducer.fit_transform(
                self.x, 
                n_neighbors=state["n_neighbors"], 
                n_MN=np.round(state["MN_ratio"] * state["n_neighbors"]).astype(np.int32), 
                n_FP=np.round(state["FP_ratio"] * state["n_neighbors"]).astype(np.int32)
            )
            t2 = time.time()
            print("time used for fit-transform: {} s".format(t2-t1))
            with open("./runs/{}/println.txt".format(self.run_name), 'a') as f:
                print("time used for fit-transform: {} s".format(t2-t1), file=f)

            # features = np.zeros((5, 1))
            # print("Please refer to image {}.".format(os.path.join(self.save_path, name)))
            # features[0] = int(input("\n1. How many clusters?\n\tcount number of clusters\n"))
            # features[1] = int(input("\n2. Rate the shape of these clusters from 1 to 4\n\t'round -> oval -> spindle-shaped -> linear'.\n"))
            # features[2] = int(input("\n3. How many 'connections' between clusters?\n\tfrom one cluster, you know what's 'next' cluster\n"))
            # features[3] = int(input("\n4. Can you observe an obvious trend or ordinal relations between clusters? Scores from 1 to 5.\n\t1 - totally not\n\t2 - not obvious\n\t3 - partly trend\n\t4 - partial trend, need imagination\n\t5 - explicit trend\n"))
            # features[4] = int(input("\n5. Do you like this visualisation? Scores from 1 to 5.\n"))
            # np.save(os.path.join(self.save_path, "{}.npy".format(name)), features.reshape(1, -1))
            
            # feedback = features[-1]



            # z = get_Ihat(normalise(z), size=100)
            # z = torch.from_numpy(z).view(1,1,100,100).float().cuda()
            # out = self.model(z).detach().cpu().view(-1)
            # feedback = torch.argmax(out) + 1            # scores in {1,2,3,4,5}

            # # r1: compared to last reward
            # if len(self.history_feedbacks)==0 or feedback > self.history_feedbacks[-1]:
            #     r1 = 1
            # else:
            #     r1 = 0

            # # r2: compared to best reward
            # r2 = feedback - self.best_feedback
            # if r2>0:
            #     self.best_feedback = feedback
            
            # print("feedback: {}, history_feedbacks: {}, r1: {}, r2: {}".format(feedback, self.history_feedbacks, r1, r2))
            # self.history_feedbacks.append(feedback.item())

            self.model.train()

            z = get_Ihat(normalise(z0), size=self.size)
            z = torch.from_numpy(z).view(1,1,self.size,self.size).float().cuda()
            
            # r1: compared to last vis
            # rigorous reward: allow false negative, rejct false positive
            out1 = []
            for _ in range(10):
                out = self.model(z, self.last_z).detach().cpu().item()
                out1.append(out)
            out1_mean = np.mean(out1)
            out1_var = np.var(out1)

            r1 = self.heuristic(out1_mean, out1_var, mode=2, t1=0.02, t2=0.5)

            # r2: compared to best vis
            out2 = []
            for _ in range(10):
                out = self.model(z, self.best_z).detach().cpu().item()
                out2.append(out)
            out2_mean = np.mean(out2)
            out2_var = np.var(out2)

            r2 = self.heuristic(out2_mean, out2_var, mode=2, t1=0.02, t2=0.5)

            # update last and best vis
            self.last_z = z
            # if r1+r2 > self.best_reward:
            if out2_mean>0.5 and out2_var<0.02:
                self.best_z = z
                self.best_z0 = z0
                self.best_name = name
                self.best_reward = r1+r2

            print("\nr1: {}, r2: {}\n".format(r1, r2))
            with open("./runs/{}/println.txt".format(self.run_name), 'a') as f:
                print("\nr1: {}, r2: {}\n".format(r1, r2), file=f)   

            plt.cla()
            draw_z(
                z=normalise(z0), 
                cls=self.label, #np.ones((z.shape[0], 1)), 
                s=1, 
                save_path=os.path.join(self.save_path, name), 
                display=False, 
                title=name + " reward: {:.4f}+{:.4f}".format(r1, r2), 
                palette='Spectral' # None
            )

            return r1 + r2

    def transition(self, action, partition):
        """
        :partition: (N, ) -- each entry indicates a cluster
        """
        
        # 1. obtain n_neighbors / MN_ratio / FP_ratio
        hp = self.combinations[action % len(self.combinations)]
        alpha, beta, gamma = hp[:, 0], hp[:, 1], hp[:, 2]       # (20,), (20,), (20,)

        alpha = {k:alpha[k] for k in range(len(alpha))}
        alpha = [alpha[i.item()] for i in partition]

        beta = {k:alpha[k] for k in range(len(beta))}
        beta = [beta[i.item()] for i in partition]

        gamma = {k:alpha[k] for k in range(len(gamma))}
        gamma = [gamma[i.item()] for i in partition]

        n_neighbors = np.round(alpha * self.current_state["n_neighbors"]).astype(np.int32)
        MN_ratio = beta * self.current_state["MN_ratio"]
        FP_ratio = gamma * self.current_state["FP_ratio"]

        # 2. obtain reward
        reward = self.obtain_reward(self.current_state)

        # 3. obtain history
        # update history actions
        self.history_actions = torch.vstack([self.history_actions, action])

        # update history rewards
        self.history_rewards = torch.hstack([self.history_rewards, torch.tensor(reward)])

        # add history info to state
        if reward > self.history_rewards[max(-self.history_len, -len(self.history_rewards))]:
            self.effect_history_actions = torch.tensor([1])
        else:
            self.effect_history_actions = torch.tensor([0])
        self.history = [self.history_actions[-self.history_len:, :], self.effect_history_actions]

        # 4. obtain state
        state = self.obtain_state(self.x, self.label, n_neighbors, MN_ratio, FP_ratio, initial=False)

        return state, reward

    def next_step(self, action, iteration, step, partition):
        self.count = self.count + 1
        self.iteration = iteration
        self.step = step

        new_state, reward = self.transition(action, partition)
        self.current_state = new_state

        terminations = 0

        done = 0 if self.count<self.num_steps else 1
        if done!=1:
            info = {}
        else:
            info = {"episode": {"r":self.best_reward, "l": self.count}}
            
        return self.current_state, reward, terminations, done, info

    def render(self):
        pass

    def reset(self):
        self.count = 0
        self.iteration = 1
        self.step = 0

        self.current_state = self.obtain_state(self.x, self.label, initial=True) # self.start
        self.reducer = myPaCMAP(
            n_components=2, 
            n_neighbors=self.current_state["n_neighbors"], 
            MN_ratio=self.current_state["MN_ratio"], 
            FP_ratio=self.current_state["FP_ratio"], 
            pair_neighbors=self.current_state["pair_neighbors"], 
            pair_MN=self.current_state["pair_MN"], 
            pair_FP=self.current_state["pair_FP"]
        )
        z0 = self.reducer.fit_transform(
            self.x, 
            n_neighbors=self.current_state["n_neighbors"], 
            n_MN=np.round(self.current_state["MN_ratio"] * self.current_state["n_neighbors"]).astype(np.int32), 
            n_FP=np.round(self.current_state["FP_ratio"] * self.current_state["n_neighbors"]).astype(np.int32)
        )
        z = get_Ihat(normalise(z0), size=self.size)
        z = torch.from_numpy(z).view(1,1,self.size,self.size).float().cuda()

        self.last_z = z
        self.best_z = z
        self.best_z0 = z0

        return self.current_state