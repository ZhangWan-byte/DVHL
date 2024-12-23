# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import gc
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.nn import GCNConv, GATv2Conv, TopKPooling
from torch.nn import MultiheadAttention

# from annoy import AnnoyIndex
# from myPaCMAP import distance_to_option
from env import DREnv

import warnings
warnings.filterwarnings('ignore')

import pickle
from copy import deepcopy
from tqdm import tqdm
from sklearn.cluster import KMeans

import torchvision as tv
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.utils import save_image

import torch.multiprocessing as mp

# from memory_profiler import profile

@dataclass
class Args:
    run_name: str = ""
    """resume previous checkpoint"""

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 3407
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Dataset
    dataset: str = "simulation"
    """simulation / sc-trans / mnist"""

    # Algorithm specific arguments
    env_id: str = "DVHL" # "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 3200 # 500000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-5 #2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1 # 4
    """the number of parallel game environments"""
    num_steps: int = 32 # 6 # 10 # 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True # False # True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 1.0 #0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 8 #10 #4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.1 # 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    num_policy: int = 6
    """the number of ensemble policies for actor-critic"""

    num_processes: int = 4
    """the number of processes for CPU parallel training"""

    # reward func
    reward_func: str = 'human-dm-surrogate'     # use ruiyuan_MLP as surrogate 
    """decision-making / human-vis / human-dm / human-dm-surrogate / ground-truth"""

    # partition-based control
    num_partition: int = 1
    """=1: only 1 partition; >1: default 20"""

    # policy network
    policy_model: str = 'MLP'
    """GNN / MLP"""

    # draw
    draw: bool = True
    """whether to draw z-imgs and save z for each step"""
    verbose: bool = False
    """whether to print prompt info"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    jianhong_advice: bool = True
    """+ norm init / + reward norm / - norm_adv=False / - clip_vloss=False"""

    agent_path: str = ""
    """agent path"""


def gauss_clusters(
    n_clusters=10, dim=10, pts_cluster=100, random_state=None, cov=1, stepsize=1,
):

    if random_state is None:
        rng = np.random.RandomState()
    else:
        rng = random_state

    n = n_clusters * pts_cluster

    s = stepsize / np.sqrt(dim)
    means = np.linspace(np.zeros(dim), n_clusters * s, num=n_clusters, endpoint=False)
    cshift_mask = np.zeros(n_clusters, dtype=bool)
    cshift_mask[15] = True
    cov = np.eye(dim) * cov

    clusters = np.array(
        [rng.multivariate_normal(m, cov, size=(pts_cluster)) for m in means]
    )

    X = np.reshape(clusters, (-1, dim))

    y = np.repeat(np.arange(n_clusters), pts_cluster)
    return X, y


def layer_init(layer, std=np.sqrt(2), bias_const=0.0, jianhong_advice=False):
    
    if jianhong_advice==False:
        torch.nn.init.orthogonal_(layer.weight, std)
    else:
        torch.nn.init.normal_(layer.weight, 0.05)

    torch.nn.init.constant_(layer.bias, bias_const)

    return layer


class GAT(torch.nn.Module):
    def __init__(self, num_node_features=50, hidden=32, num_actions=27, out_dim=1, std=1.0, history_len=7, num_partition=20, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), jianhong_advice=False):
        super().__init__()
        self.hidden = hidden
        self.num_actions = num_actions
        self.device = device
        self.edge_dim = 4
        self.history_len = history_len
        self.num_partition = num_partition
        self.out_dim = out_dim
        self.actor = 1 if out_dim==1 else 0

        self.conv1 = GATv2Conv(num_node_features, hidden, edge_dim=self.edge_dim)
        self.conv2 = GATv2Conv(hidden, hidden, edge_dim=self.edge_dim)
        self.conv3 = GATv2Conv(hidden, hidden, edge_dim=self.edge_dim)
        
        # graph feature
        self.pooling = MultiheadAttention(embed_dim=hidden, num_heads=4)

        # history feature
        self.gru = nn.GRU(input_size=self.num_actions+1, hidden_size=hidden, num_layers=2, batch_first=True)
        self.history_mlp = nn.Sequential(
            layer_init(nn.Linear(in_features=self.hidden+2, out_features=hidden), std=std, jianhong_advice=jianhong_advice),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5), 
            layer_init(nn.Linear(in_features=hidden, out_features=hidden), std=std, jianhong_advice=jianhong_advice),
            nn.LeakyReLU(), 
            nn.Dropout(p=0.5), 
            layer_init(nn.Linear(in_features=hidden, out_features=hidden), std=std, jianhong_advice=jianhong_advice),
        )

        # prediction head
        # head_out_dim = num_partition * self.out_dim if self.actor==1 else 1
        self.head = nn.Sequential(
            layer_init(nn.Linear(in_features=hidden*(num_partition+1), out_features=hidden), std=std, jianhong_advice=jianhong_advice), 
            nn.LeakyReLU(), 
            nn.Dropout(p=0.5), 
            layer_init(nn.Linear(in_features=hidden, out_features=hidden), std=std, jianhong_advice=jianhong_advice), 
            nn.LeakyReLU(), 
            nn.Dropout(p=0.5), 
            layer_init(nn.Linear(in_features=hidden, out_features=num_partition * self.out_dim), std=std, jianhong_advice=jianhong_advice)       # TODO
        )

    def forward(self, state, partition=None):

        x, edge_index, edge_attr = state["x"].float(), state["edge_index"].long(), state["edge_attr"].float()

        n, f = x.shape[0], self.hidden

        # Graph features
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv3(x, edge_index, edge_attr)

        x = self.cluster_means(x[:-1, :], partition)                                                        # (num_partition, hidden)

        x, _ = self.pooling(x, x, x)                                                                        # (num_partition, hidden)

        # History features
        history_actions, history_r1, diff_reward = state["history"]
        history_actions = history_actions[-self.history_len:]

        history_actions, _ = self.gru(F.one_hot(history_actions.flatten().long(), num_classes=self.num_actions+1).float()) # (1, hidden)
        # history_actions, _ = self.gru(history_actions)

        history_actions = torch.mean(history_actions, dim=0)

        effect = sum(history_r1[-self.history_len:])
        if effect > self.history_len*0.5:
            effect_history_actions = torch.ones([1]).to(self.device)
        else:
            effect_history_actions = torch.zeros([1]).to(self.device)

        history = torch.cat([history_actions.view(1,-1), effect_history_actions.view(1,-1), diff_reward.view(1,-1)], dim=1).float()

        history = self.history_mlp(history)

        # Prediction head
        out = torch.cat([x, history], dim=0)                                                                    # (num_partition+1, hidden)
        out = self.head(out.flatten())                                                                      # (1, num_partition*out_dim)

        out = out.view(self.num_partition, self.out_dim)                # TODO

        return out

    def cluster_means(self, x, partition):
        unique_clusters, cluster_counts = torch.unique(partition, return_counts=True)
        cluster_means = torch.zeros(len(unique_clusters), x.shape[1]).to(self.device)
        cluster_means.index_add_(0, partition.squeeze(), x)
        cluster_means /= cluster_counts.unsqueeze(1)

        return cluster_means


class my_MLP_forward_embed(nn.Module):
    """A simple MLP used for supervised learning
    In order to check the performance of the model
    and how many inputs are needed for learning-kry 24th,10,2024"""
    def __init__(self, layers, embedding_dim, input_dim, device):
        super(my_MLP_forward_embed, self).__init__()
        self.embedding_dim = embedding_dim
        self.layers = nn.ModuleList()
        self.embedx1 = nn.Linear(input_dim, embedding_dim)
        self.embedx2 = nn.Linear(embedding_dim, embedding_dim*3)
        self.embedz1 = nn.Linear(input_dim, embedding_dim)
        self.embedz2 = nn.Linear(embedding_dim, embedding_dim*3)
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.activation = nn.Sigmoid()
        
        self.device = device

    def forward(self, state, partition):

        if type(state) == type({}):
            k, m, n = state["n_neighbors"][0], state["MN_ratio"][0], state["FP_ratio"][0]
            x = torch.tensor([k, m, n, k, m, n]).view(1,-1).float().to(self.device)
        else:
            x = state        

        # print("x: ", x.shape)

        B_x = self.embedx2(self.activation(self.embedx1(x))).unsqueeze(1)
        B_z = self.embedz2(self.activation(self.embedz1(x))).unsqueeze(1)

        B_x = B_x.reshape(B_x.shape[0], 3, -1)
        B_z = B_z.reshape(B_z.shape[0], 3, -1)

        # print(B_x.shape, B_z.shape) 

        B = torch.hstack([B_x, B_z])
        x = x.unsqueeze(1)

        # print(x.shape, B.shape)

        x = 2.0 * 3.1415926 * x @ B
        x = x.squeeze(1)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

        for i in range(len(self.layers)):
            x = self.activation(self.layers[i](x))

        return x

class PolicyEnsemble(nn.Module):
    def __init__(self, num_models, num_node_features, hidden, num_actions, out_dim, std, history_len, num_partition, device, model='MLP'):
        super().__init__()
        self.base_models = nn.ModuleList([])
        for i in range(num_models):
            if model == 'GNN':
                base = GAT(
                    num_node_features, 
                    hidden, 
                    num_actions=num_actions, 
                    out_dim=out_dim, 
                    std=std, 
                    history_len=history_len-i, 
                    num_partition=num_partition, 
                    device=device
                )
            elif model == 'MLP':
                base = my_MLP_forward_embed(
                    [128,64,32,out_dim],
                    embedding_dim = int(128/2),
                    input_dim = 6, 
                    device = device
                ).to(device)
            else:
                print("wrong model name!")
                exit()

            self.base_models.append(base)

    def forward(self, state, partition=None):

        results = []
        for i, base_model in enumerate(self.base_models):
            out = base_model(state, partition)                  # (num_partition, out_dim)
            results.append(out)


        results = torch.stack(results, dim=0).mean(dim=0)       # (num_partition, out_dim)

        return results


class Agent(nn.Module):
    def __init__(self, envs, num_policy=6, num_node_features=50, hidden=64, history_len=7, num_actions=27, num_partition=None, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), actor_path=None, critic_path=None, use_multi_gpu=False, model='GNN'):
        super().__init__()

        self.critic = PolicyEnsemble(
            num_models=num_policy, 
            num_node_features=num_node_features, 
            hidden=hidden, 
            num_actions=num_actions, 
            out_dim=1, 
            std=1.0, 
            history_len=history_len, 
            num_partition=num_partition, 
            device=device,
            model=model
        )
        self.actor = PolicyEnsemble(
            num_models=num_policy, 
            num_node_features=num_node_features, 
            hidden=hidden, 
            num_actions=num_actions, 
            out_dim=num_actions, 
            std=0.01, 
            history_len=history_len, 
            num_partition=num_partition, 
            device=device,
            model=model
        )
        
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))

        self.device = device

    def get_value(self, state, partition=None):

        return self.critic(state, partition)

    def get_action_and_value(self, state, action=None, partition=None, inference=False):

        if inference:
            self.critic.eval()
            self.actor.eval()
        else:
            self.critic.train()
            self.actor.train()


        if type(state)==type([]):
            ac = []
            pbs = []
            ent = []
            value = []
            for i in range(len(state)):
                logits = self.actor(state[i], partition)
                # print("logits: ", logits)
                probs = Categorical(logits=logits)
                if action is None:
                    action = probs.sample().view(-1)
                    ac.append(action)
                    pbs.append(probs.log_prob(action))
                else:
                    ac.append(action[i])
                    pbs.append(probs.log_prob(action[i]))
                ent.append(probs.entropy())
                value.append(self.critic(state[i], partition))

            ac = torch.vstack(ac)
            pbs = torch.vstack(pbs)
            ent = torch.vstack(ent)
            value = torch.vstack(value)
            # print("probs: ", pbs)
            return ac, pbs, ent, value
        else:
            logits = self.actor(state, partition)
            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            
            return action, probs.log_prob(action), probs.entropy(), self.critic(state, partition)

def get_partition(data, k=20, labels=None):
    if labels==None:
        kms = KMeans(n_clusters=k).fit(data)
        kms_labels = kms.labels_

        mean_var = np.array([[i.mean(), i.var()] for i in kms.cluster_centers_])
        ind = np.lexsort((mean_var[:, 1], mean_var[:, 0]))
        label_trans = {key: value for key, value in list(zip(np.arange(20), ind))}

        invariant_partition = torch.tensor([label_trans[i] for i in kms_labels])

        return invariant_partition

    else:
        print("not implemented in `get_partition`")
        exit()


# @profile
def main():
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)                   # 1 * 32                1 * 8
    args.minibatch_size = int(args.batch_size // args.num_minibatches)      # 32 // 8 = 4           8 // 4 = 2
    args.num_iterations = args.total_timesteps // args.batch_size           # [3200/32]=100         [1600/8]=200
    
    print("num_iterations: {}, minibatch_size: {}, num_steps: {}".format(
        args.num_iterations, args.minibatch_size, args.num_steps))

    if args.jianhong_advice==True:
        args.norm_adv = False
        args.clip_vloss = False

    if args.run_name=="":
        run_name = f"{args.env_id}__{args.exp_name}__{time.strftime('%m%d%H%M%S', time.localtime())}__{args.seed}"
    else:
        run_name = args.run_name

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    # pickle.dump(open(f"runs/{run_name}/args.pkl", "w"), args)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    use_multi_gpu = True if torch.cuda.device_count()>1 else False
    print("device: ", device)
    print("device num: ", torch.cuda.device_count())

    # data generation

    # 1. simulation
    if args.dataset=='simulation':
        data, labels = gauss_clusters(
            n_clusters=20,
            dim=50,
            pts_cluster=1000,
            stepsize=6,
            random_state=None,
        )
    # 2. single-cell transcriptomics
    elif args.dataset=='sc-trans':
        data = np.load("../DVHL_others/exp2/data.npy")
        labels = np.load("../DVHL_others/exp2/labels.npy")
    # 3. MNIST
    elif args.dataset=='mnist':
        data = np.load("./data/MNIST/X_train.npy")
        labels = np.load("./data/MNIST/y_train.npy")
    else:
        print("wrong dataset!")
        exit()
    
    idx = np.random.choice(data.shape[0], 1000, replace=False)
    data, labels = data[idx], labels[idx]

    np.save("./runs/{}/data_online_1k.npy".format(run_name), data)
    np.save("./runs/{}/labels_online_1k.npy".format(run_name), labels)

    # partition-based control
    if args.num_partition == 1:
        num_partition = 1
        partition = torch.zeros(data.shape[0]).int().to(device)
    else:
        num_partition = len(np.unique(labels))
        if args.dataset=='simulation':
            partition = get_partition(data, k=num_partition, labels=None).to(device)           # (data.shape[0], ) -- each entry is a cluster
        elif args.dataset=='sc-trans':
            partition = torch.from_numpy(labels).to(device)
        elif args.dataset=='mnist':
            partition = get_partition(data, k=num_partition, labels=None).to(device)
        else:
            print("wrong dataset!")
            exit()

    # 2. MNIST
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # trainset = tv.datasets.MNIST(root='./data',  train=True, download=False, transform=transform)
    # traindata = [i[0].unsqueeze(0) for i in trainset]
    # trainlabel = [i[1] for i in trainset]
    # data = torch.vstack(traindata).numpy().reshape(60000, 28*28)
    # labels = torch.tensor(trainlabel).numpy().reshape(60000, 1)
    # idx = np.random.choice(data.shape[0], 10000, replace=False)
    # data, labels = data[idx], labels[idx]

    print("data: {}, labels: {}".format(data.shape, labels.shape))
    with open("./runs/{}/println.txt".format(run_name), 'a') as f:
        print("data: {}, labels: {}".format(data.shape, labels.shape), file=f)
    envs = DREnv(
        data.astype('float32'), 
        labels.astype('float32'), 
        model_path="./exp1/model_online.pt",  
        action_space=27, 
        history_len=7, 
        save_path=f"./runs/{run_name}", 
        num_steps = args.num_steps, 
        num_partition=num_partition, 
        run_name=run_name, 
        device=device, 
        reward_func=args.reward_func, 
        draw=args.draw, 
        verbose=args.verbose, 
        dataset=args.dataset
    )

    agent = Agent(
        envs, 
        num_policy=args.num_policy, 
        num_node_features=data.shape[1], 
        hidden=32, 
        history_len=7, 
        num_actions=27, 
        num_partition=num_partition, 
        use_multi_gpu=use_multi_gpu, 
        device=device,
        model=args.policy_model
    ).to(device)
    if args.agent_path != "":
        agent.load_state_dict(torch.load(args.agent_path))
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    print("actor params: {}".format(sum([p.numel() for p in agent.actor.parameters()])))
    print("critic params: {}".format(sum([p.numel() for p in agent.critic.parameters()])))
    with open("./runs/{}/println.txt".format(run_name), 'a') as f:
        print("actor params: {}".format(sum([p.numel() for p in agent.actor.parameters()])), file=f)
        print("critic params: {}".format(sum([p.numel() for p in agent.critic.parameters()])), file=f)

    print("anneal_lr: {}".format(args.anneal_lr))
    with open("./runs/{}/println.txt".format(run_name), 'a') as f:
        print("anneal_lr: {}".format(args.anneal_lr), file=f)

    # ALGO Logic: Storage setup
    actions_0 = torch.zeros((args.num_steps, num_partition, args.num_envs)).to(device)
    logprobs_0 = torch.zeros((args.num_steps, num_partition, args.num_envs)).to(device)
    rewards_0 = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones_0 = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_0 = torch.zeros((args.num_steps, num_partition, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()

    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # reset obs each iteration, as other variables do. otherwise OOM.
        
        next_obs = envs.reset()

        obs = []        

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        terminate_iter = False

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            # obs[step] = next_obs
            obs.append(next_obs)
            dones_0[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():

                cnt = 0
                # handle accident or illegal situation
                while True:
                    action, logprob, _, value = agent.get_action_and_value(next_obs, partition=partition, inference=True)
                    print("action idx: ", action)
                    hp = envs.combinations[action.detach().cpu() % len(envs.combinations)]
                    hp = hp.reshape(-1,3)
                    print("action: ", hp)
                    alpha, beta, gamma = hp[:, 0], hp[:, 1], hp[:, 2]

                    if args.reward_func != 'human-dm-surrogate' or args.reward_func != 'ground-truth':
                        alpha = {k:alpha[k] for k in range(len(alpha))}
                        alpha = [alpha[i.item()] for i in partition]

                        beta = {k:alpha[k] for k in range(len(beta))}
                        beta = [beta[i.item()] for i in partition]

                        gamma = {k:alpha[k] for k in range(len(gamma))}
                        gamma = [gamma[i.item()] for i in partition]

                        print(alpha[0], beta[0], gamma[0])

                    n_neighbors = np.round(alpha * envs.current_state["n_neighbors"]).astype(np.int32)
                    MN_ratio = beta * envs.current_state["MN_ratio"]
                    FP_ratio = gamma * envs.current_state["FP_ratio"]

                    print("n, MN, FP: ", n_neighbors[0], MN_ratio[0], FP_ratio[0])

                    if args.cuda:
                        torch.cuda.empty_cache()
                    gc.collect()

                    cnt += 1
                    if cnt > 3:
                        terminate_iter = True
                        break

                    if np.min(n_neighbors) < 1 or np.min(MN_ratio*n_neighbors) < 1 or np.min(FP_ratio*n_neighbors) < 1:
                        continue         
                    else:
                        break

            if terminate_iter == True:
                print("================== terminate_iter is True ==================")
                break
            else:
                values_0[step] = value.flatten().view(-1,1)
                actions_0[step] = action.view(-1,1)
                logprobs_0[step] = logprob.view(-1,1)

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, done, infos = envs.next_step(action, iteration, step, partition)
                # rewards[step] = torch.tensor(reward).to(device).view(-1)
                # next_done = torch.tensor(next_done).to(device)
                next_done = done

                # print("\nhistory_len:{}\nhistory_rewards:{}\nhistory_actions:{}\nreward:{}\nnext_done:{}".format(
                #     envs.history_len, envs.history_rewards, envs.history_actions, reward, 
                #     next_done))

                # with open("./runs/{}/println.txt".format(run_name), 'a') as f:
                #     print("\nhistory_len:{}\nhistory_rewards:{}\nhistory_actions:{}\nreward:{}\nnext_done:{}".format(
                #     envs.history_len, envs.history_rewards, envs.history_actions, reward, 
                #     next_done), file=f)
                
                # print("best: {}".format(envs.best_name))
                
                # with open("./runs/{}/println.txt".format(run_name), 'a') as f:
                #     print("best: {}".format(envs.best_name), file=f)

                if "episode" in infos.keys():
                    print(f"global_step={global_step}, episodic_return={infos['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_step)

            with open("./runs/{}/println.txt".format(run_name), 'a') as f:
                    print("iter{}_step{}\tcurrent:{}\tbest:{}\tr1:{}\tr2:{}\tr:{}".format(
                        iteration, step, envs.features, envs.best_feats, envs.r1, envs.r2, reward), file=f)
            print("iter{}_step{}\tcurrent:{}\tbest:{}\taction:{}\tr1:{}\tr2:{}\tr:{}\n\n".format(
                        iteration, step, envs.features, envs.best_feats, action, envs.r1, envs.r2, reward))

        with open("./runs/{}/best_params.txt".format(run_name), 'a') as f:
            print("{}\t{}\t{}\t{}".format(
                iteration, 
                envs.best_feats[0].item(), 
                envs.best_feats[1].item(), 
                envs.best_feats[2].item()
                ), file=f)

        if args.cuda:
            torch.cuda.empty_cache()
        gc.collect()

        # truncate redundant null steps
        if terminate_iter==True:
            actions = actions_0[:step]
            logprobs = logprobs_0[:step]
            values = values_0[:step]
            rewards = rewards_0[:step]
            dones = dones_0[:step]
            dones[-1] = 1

            actual_num_steps = step
            actual_batch_size = step

            writer.add_scalar("charts/episodic_return", sum(envs.history_r1[-(envs.step+1):]).item(), global_step)
            writer.add_scalar("charts/episodic_length", envs.count, global_step)

        else:
            actions = actions_0
            logprobs = logprobs_0
            values = values_0
            rewards = rewards_0
            dones = dones_0

            actual_num_steps = args.num_steps
            actual_batch_size = args.batch_size

        if args.jianhong_advice==True:
            rewards = (rewards - rewards.mean(dim=0)) / (rewards.std(dim=0) + 1e-8)

        if args.reward_func == 'decision-making':
            # save agent with BEST single mse
            if min(envs.history_mse[-actual_num_steps:]) < envs.best_mse:
                torch.save(agent.state_dict(), "./runs/{}/best_mse_agent.pt".format(run_name))
                envs.best_mse = min(envs.history_mse[-actual_num_steps:])
        elif args.reward_func in ['human-dm', 'human-dm-surrogate', 'ground-truth']:
            # save agent with BEST episodic accumulative rewards
            if sum(envs.history_rewards[-actual_num_steps:]) > envs.best_reward:
                torch.save(agent.state_dict(), "./runs/{}/best_rewards_agent.pt".format(run_name))
                envs.best_reward = sum(envs.history_rewards[-actual_num_steps:])
        else:
            pass

        # bootstrap value if not done
        with torch.no_grad():
            next_values = agent.get_value(next_obs, partition).reshape(1, -1)           # (1, num_partition)

            advantages = torch.zeros((actual_num_steps, num_partition)).to(device)
            lastgaelam = 0
            for t in reversed(range(actual_num_steps)):
                if t == actual_num_steps - 1:
                    nextnonterminal = 1.0 - dones[-1]
                    nextvalues = next_values.view(-1)
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1].view(-1)
                delta = rewards[t].view(-1) + args.gamma * nextvalues * nextnonterminal - values[t].view(-1)        # (num_partition, )
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam

            returns = advantages + values.squeeze()           # (num_steps, num_partition)

        b_obs = obs #deepcopy(obs)
        b_logprobs = logprobs.reshape(actual_num_steps, -1)
        b_actions = actions.reshape(actual_num_steps, -1)
        b_values = values.reshape(actual_num_steps, -1)

        b_advantages = advantages.reshape(actual_num_steps, -1)
        b_returns = returns.reshape(actual_num_steps, -1)

        # Optimizing the policy and value network
        b_inds = np.arange(actual_batch_size)
        clipfracs = []
            
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, actual_batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                b_obs_i = [b_obs[i] for i in mb_inds]
                b_actions_i = b_actions.long()[mb_inds]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs_i, b_actions_i, partition=partition, inference=False)
                # print("newlogprob: {}, b_logprobs[mb_inds]: {}, mb_inds: {}".format(newlogprob, b_logprobs[mb_inds], mb_inds))
                # with open("./runs/{}/println.txt".format(run_name), 'a') as f:
                #     print("newlogprob: {}, b_logprobs[mb_inds]: {}, mb_inds: {}".format(newlogprob, b_logprobs[mb_inds], mb_inds), file=f)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                # print("logratio: ", logratio)
                # with open("./runs/{}/println.txt".format(run_name), 'a') as f:
                #     print("logratio: ", logratio, file=f)
                
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean(dim=0)
                    approx_kl = ((ratio - 1) - logratio).mean(dim=0)
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean(dim=0).detach().cpu().numpy()]

                mb_advantages = b_advantages[mb_inds]
                # print("mb_advantages: ", mb_advantages)
                # with open("./runs/{}/println.txt".format(run_name), 'a') as f:
                #     print("mb_advantages: ", mb_advantages, file=f)
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean(dim=0)) / (mb_advantages.std(dim=0) + 1e-8)
                # print("mb_advantages, ratio: ", mb_advantages, ratio)
                # with open("./runs/{}/println.txt".format(run_name), 'a') as f:
                #     print("mb_advantages, ratio: ", mb_advantages, ratio, file=f)
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean(dim=0)

                # Value loss
                newvalue = newvalue.view(-1, num_partition)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean(dim=0)
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean(dim=0)

                entropy_loss = entropy.mean(dim=0)
                # print("sub-loss shapes", pg_loss.shape, entropy_loss.shape, v_loss.shape) # (20,) (20,) (20,)
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
                print("loss: ", loss, pg_loss, entropy_loss, v_loss)
                # with open("./runs/{}/println.txt".format(run_name), 'a') as f:
                #     print("loss: ", loss, pg_loss, entropy_loss, v_loss, file=f)
                optimizer.zero_grad()
                loss.mean().backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.detach().cpu().numpy(), b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.mean().item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.mean().item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.mean().item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.mean().item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.mean().item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/rewards", envs.history_rewards[-actual_num_steps:].mean().item(), global_step)
        if args.reward_func == 'decision-making':
            writer.add_scalar("losses/mse", min(np.array(envs.history_mse)[-actual_batch_size:]), global_step)

        now_time = time.time()
        # print("SPS:", int(global_step / (now_time - start_time)))
        # with open("./runs/{}/println.txt".format(run_name), 'a') as f:
        #     print("SPS:", int(global_step / (now_time - start_time)), file=f)
        writer.add_scalar("charts/SPS", int(global_step / (now_time- start_time)), global_step)

        # save agent with BEST episodic accumulative reward
        if torch.sum(envs.history_rewards[-actual_num_steps:]) > envs.best_epoch_reward:
            torch.save(agent.state_dict(), "./runs/{}/best_epoch_agent.pt".format(run_name))
            envs.best_epoch_reward = torch.sum(envs.history_rewards[-actual_num_steps:])

        torch.save(agent.state_dict(), "./runs/{}/agent.pt".format(run_name))
        torch.save(envs.history_rewards.detach().cpu(), "./runs/{}/history_rewards.pt".format(run_name))
        torch.save(envs.history_actions.detach().cpu(), "./runs/{}/history_actions.pt".format(run_name))
        if args.reward_func == 'decision-making':
            np.save("./runs/{}/history_mse.npy".format(run_name), np.array(envs.history_mse))

        if args.cuda:
            torch.cuda.empty_cache()
        gc.collect()

        # envs.update_surrogate(iteration)

    envs.close()
    writer.close()



if __name__ == "__main__":
    main()