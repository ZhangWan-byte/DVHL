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

import pickle

import networkx as nx
from scipy.spatial.distance import pdist, squareform

from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import mean_squared_error

from compute_feature import generate_features

# from memory_profiler import profile

def get_tree(data):
    n, dim = data.shape
    tree = AnnoyIndex(dim, metric='euclidean')
    for i in range(n):
        tree.add_item(i, data[i, :])
    tree.build(20)

    return tree


def generate_points_on_segment(A, B, num_points):
    # Extract coordinates of points A and B
    x1, y1 = A
    x2, y2 = B
    
    # Calculate the increment for each coordinate
    delta_x = (x2 - x1) / (num_points + 1)
    delta_y = (y2 - y1) / (num_points + 1)
    
    # Generate points on the segment
    points = []
    for i in range(1, num_points + 1):
        x = x1 + i * delta_x
        y = y1 + i * delta_y
        points.append((x, y))
    
    return points


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
    def __init__(self, x, label, model_path="./exp1/model_online.pt", action_space=27, history_len=7, save_path=None, num_steps=32, num_partition=20, size=256, run_name=None, inference=False, data=None, labels=None, idx=None, r3_coef=3, device=torch.device('cpu'), reward_func='decision-making', draw=True, verbose=False):
        self.x = x
        self.label = label

        self.draw = draw
        self.verbose = verbose

        self.name = None
        self.best_name = None
        self.size = size

        self.data = data
        self.labels = labels
        self.idx = idx

        self.run_name = run_name
        self.inference = inference

        self.count = 0
        self.current_state = None
        self.action_space = action_space

        self.device = device
        self.done_list = [torch.tensor(0).to(self.device), torch.tensor(1).to(self.device)]
        self.zero = torch.tensor([0.]).to(self.device)
        self.one = torch.tensor([1.]).to(self.device)
        self.reward = torch.tensor([0.]).to(self.device)

        self.num_partition = num_partition

        # objective surrogate
        self.reward_func = reward_func
        
        if reward_func == 'human-vis':
            self.model = SiameseNet(
                hidden=256, #64, 
                block=BasicBlock, 
                num_block=[3,4,6,3], #[1,1,1,1], 
                in_channels=1, 
                out_channels=[10, 16, 24, 32]
            ).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("human surrogate params: ", sum([p.numel() for p in self.model.parameters()]))
            with open("./runs/{}/println.txt".format(self.run_name), 'a') as f:
                print("human surrogate params: ", sum([p.numel() for p in self.model.parameters()]), file=f)
            self.model.train()

            self.out1 = torch.zeros((10, 1)).to(self.device)
            self.out2 = torch.zeros((10, 1)).to(self.device)

            self.coef = r3_coef
            self.last_length_reward = torch.tensor([0.]).to(self.device)
            self.history_lengths = torch.tensor([]).to(self.device)

        elif reward_func == 'decision-making':
            self.best_mse = np.inf
            self.last_mse = np.inf

            self.history_mse = []

        elif reward_func == 'human-dm':
            self.best_feats = np.ones((1, 14))
            self.last_feats = np.ones((1, 14))

        else:
            pass

        # actions
        if self.inference==False:
            self.alpha_values = [0.8, 1.0, 1.2]                # value range of ratio of kNN
            self.beta_values = [0.8, 1.0, 1.2]                 # value range of ratio of mid-pairs
            self.gamma_values = [0.8, 1.0, 1.2]                # value range of ratio of negatives
            # self.hetero_homo = [0, 1, 2]                     # 0 - random / 1 - hetero / 2 - homo
        else:
            self.alpha_values = [0.5, 1.0, 2.0]                # value range of ratio of kNN
            self.beta_values = [0.5, 1.0, 2.0]                 # value range of ratio of mid-pairs
            self.gamma_values = [0.5, 1.0, 2.0]                # value range of ratio of negatives
        
        # 3 * 3 * 3 = 27 actions (0,1,...,26)
        self.combinations = np.array(
            list(itertools.product(self.alpha_values, self.beta_values, self.gamma_values))
        )

        # best & last vis
        self.best_z = None
        self.best_z0 = None
        self.last_z = None

        # record during training
        self.best_epoch_reward = -1000000
        self.history_rewards = torch.tensor([0.]).to(self.device)
        self.history_r1 = torch.tensor([]).to(self.device)

        # history: actions / effect
        self.history_len = history_len
        self.history_actions = torch.zeros((history_len, self.num_partition)).to(self.device)
        self.diff_reward = torch.zeros([1]).to(self.device)
        # self.effect_history_actions = torch.zeros([1]).to(self.device)

        # like a sentence, abcdefg0 / fdsjkhy1, each character is one_hot
        self.history = [self.history_actions, self.history_r1, self.diff_reward]
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

            n_neighbors = np.round(np.ones(x.shape[0]) * 20).astype(np.int32) #None
            MN_ratio = np.ones(x.shape[0]) * 1.0 #0.5 #2.0 # 0.5
            FP_ratio = np.ones(x.shape[0]) * 5.0 #1.0 #20.0 # 2.0

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
                    r = max((out_mean - torch.sqrt(out_var)*3).round(), self.zero)
                else:
                    r = min((out_mean + torch.sqrt(out_var)*3).round(), self.one)

        elif mode==1:
        # constructively conservative and continuous and dense
            if out_var < t1:
                r = out_mean
            else:
                if out_mean > t2:
                    r = max((out_mean - torch.sqrt(out_var)), self.zero)
                else:
                    r = min((out_mean + torch.sqrt(out_var)), self.one)

        elif mode==2:
        # conservative and continuous and dense
            r = max((out_mean - torch.sqrt(out_var)), self.zero)

        elif mode==3:
        # conservative and discrete and sparse
            r = max((out_mean - torch.sqrt(out_var)*3).round(), self.zero)

        elif mode==4:
        # directly use prob as reward
            r = out_mean

        else:
            print("not implemented heuristic!")
            exit()

        return r

    def MST_length(self, z, k=5):

        A = kneighbors_graph(normalise(z), n_neighbors=k, mode='distance', metric='euclidean', include_self=False)

        G = nx.Graph(A)

        MST = nx.minimum_spanning_tree(G)
        length = sum(weight for _, _, weight in MST.edges(data='weight'))

        return length

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
            # print("\n\n{} fit-transforming...".format(name))
            # with open("./runs/{}/println.txt".format(self.run_name), 'a') as f:
            #     print("\n\n{} fit-transforming...".format(name), file=f)

            t1 = time.time()
            z0 = self.reducer.fit_transform(
                self.x, 
                n_neighbors=state["n_neighbors"], 
                n_MN=np.round(state["MN_ratio"] * state["n_neighbors"]).astype(np.int32), 
                n_FP=np.round(state["FP_ratio"] * state["n_neighbors"]).astype(np.int32)
            )
            t2 = time.time()

            if self.verbose:
                print("time used for fit-transform: {} s".format(t2-t1))
                with open("./runs/{}/println.txt".format(self.run_name), 'a') as f:
                    print("time used for fit-transform: {} s".format(t2-t1), file=f)

            if self.draw:
                torch.save(torch.from_numpy(z0), os.path.join(self.save_path, "z_{}.pt".format(name)))
                print("z saved to: {}".format(os.path.join(self.save_path, "z_{}.pt".format(name))))
                with open("./runs/{}/println.txt".format(self.run_name), 'a') as f:
                    print("z saved to: {}".format(os.path.join(self.save_path, "z_{}.pt".format(name))), file=f)
            
            # get reward
            if self.reward_func == 'human-vis':

                self.model.train()

                z = get_Ihat(normalise(z0), size=self.size)
                z = torch.from_numpy(z).view(1,1,self.size,self.size).float().to(self.device)

                # r1: compared to last vis
                for out_idx in range(10):
                    out = self.model(z, self.last_z)
                    self.out1[out_idx] = out
                r1 = self.heuristic(out_mean=torch.mean(self.out1), out_var=torch.var(self.out1), mode=2, t1=0.02, t2=0.5)
                # e.g., tensor([0.2280], device='cuda:0')

                # r2: compared to best vis
                for out_idx in range(10):
                    out = self.model(z, self.best_z)
                    self.out2[out_idx] = out
                r2 = self.heuristic(out_mean=torch.mean(self.out2), out_var=torch.var(self.out2), mode=2, t1=0.02, t2=0.5)
                # e.g., tensor([0.2280], device='cuda:0')

                # r3: MST length
                length = self.MST_length(z0)
                length_reward = torch.tensor([self.coef / length]).to(self.device)
                r3 = length_reward - self.last_length_reward
                
                self.history_lengths = torch.hstack([self.history_lengths, length_reward])

                r = r1 + r2 + r3

                # update last
                self.last_length_reward = length_reward
                self.last_z = z

                # update best
                if torch.mean(self.out2)>0.5 and torch.var(self.out2)<0.02:
                # if r1>0 and r2>0:
                    self.best_z = z
                    self.best_z0 = z0
                    self.best_name = name

                reward_info = " reward: {:.4f}+{:.4f}+{:.4f}".format(r1.item(), r2.item(), r3.item())

                # print("\nr1: {}, r2: {}, r3: {}\n".format(r1, r2, r3))
                # with open("./runs/{}/println.txt".format(self.run_name), 'a') as f:
                #     print("\nr1: {}, r2: {}, r3: {}\n".format(r1, r2, r3), file=f)

            elif self.reward_func == 'decision-making':
                z = normalise(z0)

                agc = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5)
                agc.fit(z)

                # scaffold extraction
                points = []
                for i in range(len(np.unique(agc.labels_))):
                    cluster = z[agc.labels_==i]

                    pca = PCA(n_components=1)
                    pca.fit(cluster)

                    components = pca.components_
                    long_axis = components[0]

                    mean_point = np.mean(cluster, axis=0)
                    distance_matrix = squareform(pdist(cluster))
                    major_axis_length = np.max(distance_matrix)
                    endpoint1 = mean_point - long_axis * (major_axis_length / 2)
                    endpoint2 = mean_point + long_axis * (major_axis_length / 2)

                    points.append(endpoint1)
                    points.append(endpoint2)
                    points.extend(generate_points_on_segment(endpoint1, endpoint2, num_points=4))
                points = np.vstack(points)

                # curve fitting
                # 1. rotate to x-axis
                # 2. sort based on x
                points = points[np.argsort(points[:, 0])]
                # 3. normalise
                points = normalise(points)

                # 4. fitting
                poly = PolynomialFeatures(degree=100, include_bias=False)
                poly_features = poly.fit_transform(points[:, 0].reshape(-1, 1))

                poly_reg = LinearRegression()
                poly_reg.fit(poly_features, points[:, 1])

                # 5. evaluation
                x = points[:, 0].reshape(-1,1)
                y_pred = poly_reg.predict(poly.transform(x))

                mse = mean_squared_error(points[:, 1], y_pred)

                self.history_mse.append(mse)

                # r1: compared to last
                r1 = self.last_mse - mse

                # r2: compared to best
                r2 = self.best_mse - mse

                # r = r1 + r2
                r = r1

                # update last
                self.last_mse = mse

                # update best
                if r2>0:
                    self.best_mse = mse
                    self.best_name = name

                # reward_info = " reward: {:.4f}+{:.4f}".format(r1, r2)
                reward_info = " reward: {:.4f}".format(r1)
                    
            elif self.reward_func == 'human-dm':
                z = normalise(z0)

                agc = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5)
                agc.fit(z)

                features = generate_features(z=z, labels=agc.labels_, data=self.x).reshape(1,-1)

                feat1 = np.hstack([self.last_feats, features]).reshape(1,-1)
                feat2 = np.hstack([self.best_feats, features]).reshape(1,-1)
                
                r1 = self.lgb.predict(feat1.reshape(1,-1)).item()

                r2 = self.lgb.predict(feat1.reshape(1,-1)).item()

                r = r1 + r2

                # update last
                self.last_feats = features

                # update best
                if r2>0:
                    self.best_feats = features
                    self.best_name = name

                reward_info = " reward: {:.4f}+{:.4f}".format(r1, r2)
                

            else:
                print("wrong reward_func!")
                exit()

            # update history r1
            self.history_r1 = torch.hstack([self.history_r1, torch.tensor(r1).to(self.device)])

            if self.draw:

                plt.cla()
                draw_z(
                    z=normalise(z0), 
                    cls=self.label, #np.ones((z.shape[0], 1)), 
                    s=1, 
                    save_path=os.path.join(self.save_path, name), 
                    display=False, 
                    title=name + reward_info, 
                    palette='Spectral' # None
                )
                print("img saved to {}".format(os.path.join(self.save_path, name)))

            # if self.inference:
            #     tree = get_tree(self.x)

            #     knn = 2
            #     entire_z = np.zeros((self.data.shape[0], knn))

            #     for i in range(self.x.shape[0]):
            #         entire_z[self.idx[i]] = z0[i]

            #     for i in range(self.data.shape[0]):
            #         if i in self.idx:
            #             continue
            #         else:
            #             ids = tree.get_nns_by_vector(self.data[i], knn)
            #             low_dim = np.mean(z0[ids], axis=0)
            #         entire_z[i] = low_dim

            #     draw_z(
            #         z=normalise(entire_z), 
            #         cls=self.labels, 
            #         s=1, 
            #         save_path=os.path.join(self.save_path, "20k_iter{}_step{}".format(self.iteration, self.step)), 
            #         display=False, 
            #         title="20k_iter{}_step{}".format(self.iteration, self.step) + " reward: {:.4f}+{:.4f}".format(r1, r2), 
            #         palette='Spectral'
            #     )
            #     print("img saved to {}".format(os.path.join(self.save_path, "20k_iter{}_step{}".format(self.iteration, self.step))))

            return r

    def transition(self, action, partition):
        """
        :partition: (N, ) -- each entry indicates a cluster
        """
        
        # 1. obtain n_neighbors / MN_ratio / FP_ratio
        hp = self.combinations[action.detach().cpu() % len(self.combinations)]
        alpha, beta, gamma = hp[:, 0], hp[:, 1], hp[:, 2]       # (20,), (20,), (20,)
        # print("alpha: ", alpha)
        # print("beta: ", beta)
        # print("gamma: ", gamma)
        alpha = {k:alpha[k] for k in range(len(alpha))}
        alpha = [alpha[i.item()] for i in partition]

        beta = {k:beta[k] for k in range(len(beta))}
        beta = [beta[i.item()] for i in partition]

        gamma = {k:gamma[k] for k in range(len(gamma))}
        gamma = [gamma[i.item()] for i in partition]

        n_neighbors = np.round(alpha * self.current_state["n_neighbors"]).astype(np.int32)
        MN_ratio = beta * self.current_state["MN_ratio"]
        FP_ratio = gamma * self.current_state["FP_ratio"]
        
        # 2. obtain state - others
        state = self.obtain_state(self.x, self.label, n_neighbors, MN_ratio, FP_ratio, initial=False)
        
        # 3. obtain reward
        self.reward = torch.tensor([self.obtain_reward(state)]).to(self.device)

        # 4. obtain state - history

        self.history_actions = torch.vstack([self.history_actions, action+1])                                       # update history actions

        self.history_rewards = torch.hstack([self.history_rewards, self.reward])                                    # update history rewards

        # if sum(self.history_r1[-min(self.step+1, self.history_len):]) > 0:                                     # add history info to state
        #     self.effect_history_actions = torch.tensor([1]).to(self.device)
        # else:
        #     self.effect_history_actions = torch.tensor([0]).to(self.device)

        self.diff_reward = self.reward - self.history_rewards[-2]

        self.history = [self.history_actions[-min(self.step+1, self.history_len):, :], self.history_r1, self.diff_reward]

        state["history"] = [i.to(self.device) for i in self.history]

        return state, self.reward

    def next_step(self, action, iteration, step, partition):
        self.count = self.count + 1
        self.iteration = iteration
        self.step = step

        new_state, reward = self.transition(action, partition)
        self.current_state = new_state
        terminations = 0

        done = self.done_list[0] if self.count<self.num_steps else self.done_list[1]
        if done!=1:
            info = {}
        else:
            info = {
                "episode": {
                    "r":sum(self.history_r1[-(self.step+1):]).item(), 
                    "l": self.count
                }
            }
            
        return new_state, reward, terminations, done, info

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

        if self.reward_func == 'human-vis':
            z = get_Ihat(normalise(z0), size=self.size)
            z = torch.from_numpy(z).view(1,1,self.size,self.size).float().to(self.device)

            self.last_z = z
            self.best_z = z
            self.best_z0 = z0

            length = self.MST_length(z0)
            self.last_length_reward = torch.tensor([self.coef / length]).to(self.device)
        
        elif self.reward_func == 'decision-making':
            z = normalise(z0)

            agc = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5)
            agc.fit(z)

            # scaffold extraction
            points = []
            for i in range(len(np.unique(agc.labels_))):
                cluster = z[agc.labels_==i]

                pca = PCA(n_components=1)
                pca.fit(cluster)

                components = pca.components_
                long_axis = components[0]

                mean_point = np.mean(cluster, axis=0)
                distance_matrix = squareform(pdist(cluster))
                major_axis_length = np.max(distance_matrix)
                endpoint1 = mean_point - long_axis * (major_axis_length / 2)
                endpoint2 = mean_point + long_axis * (major_axis_length / 2)

                points.append(endpoint1)
                points.append(endpoint2)
                points.extend(generate_points_on_segment(endpoint1, endpoint2, num_points=8))
            points = np.vstack(points)

            # curve fitting
            # 1. rotate to x-axis
            # 2. sort based on x
            points = points[np.argsort(points[:, 0])]
            # 3. normalise
            points = normalise(points)

            # 4. fitting
            poly = PolynomialFeatures(degree=100, include_bias=False)
            poly_features = poly.fit_transform(points[:, 0].reshape(-1, 1))

            poly_reg = LinearRegression()
            poly_reg.fit(poly_features, points[:, 1])

            # 5. evaluation
            x = points[:, 0].reshape(-1,1)
            y_pred = poly_reg.predict(poly.transform(x))

            mse = mean_squared_error(points[:, 1], y_pred)

            self.last_mse = mse
            self.best_mse = mse

        elif self.reward_func == 'human-dm':

            z = normalise(z0)

            agc = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5)
            agc.fit(z)

            features = generate_features(z=z, labels=agc.labels_, data=self.x)

            self.best_feats = np.ones((1,14))
            self.last_feats = np.ones((1,14))

            self.lgb = pickle.load(open("./lgb.pkl", "rb"))

        else:
            print("wrong reward_func!")
            exit()

        if self.draw:
            plt.cla()
            draw_z(
                z=normalise(z0), 
                cls=self.label, #np.ones((z.shape[0], 1)), 
                s=1, 
                save_path=os.path.join(self.save_path, "initial"), 
                display=False, 
                title="initial", 
                palette='Spectral' # None
            )
            print("img saved to {}".format(os.path.join(self.save_path, "initial")))

        return self.current_state