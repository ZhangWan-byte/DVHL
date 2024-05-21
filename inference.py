import os
import gc
import tyro
import time
import random
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from ppo import *
from env import *

from torch.cuda.amp import autocast, GradScaler

@dataclass
class Args:
    run_name: str = ""
    """resume previous checkpoint"""

    exp_name: str = "ppo" # os.path.basename(__file__)[:-len(".py")]
    """the name of this experiment"""
    seed: int = 3407
    """seed of the experiment"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    num_envs: int = 1
    """number of environments"""
    actor_path: str = ""
    """path to load pre-trained actor"""

    # Algorithm specific
    env_id: str = "DVHL"
    """the id of the environment"""
    num_steps: int = 32 #100
    """number of rollout steps"""
    num_policy: int = 6
    """number of policies"""
    learning_rate: float = 4e-3
    """the learning rate of the optimizer"""
    search: str = "sampling" # random / epsilon-greedy / sampling
    """types of search algorithm"""
    param_lambda: float = 0.1
    """weights for imitation learning loss"""
    update: bool = True
    """whether to fine-tune policy"""
    subset: int = 0
    """whether to use only a subset of entire dataset"""
    use_base_selection: bool = False
    """whether to select best results among all base policies"""

    # Dataset
    dataset: str = "sc-trans"
    """simulation / sc-trans"""

    # reward func
    reward_func: str = 'real-human'
    """decision-making / human-vis / human-dm / real-human"""

    # draw
    draw: bool = True
    """whether to draw z-imgs and save z for each step"""
    verbose: bool = True
    """whether to print prompt info"""


class InferenceAgent(nn.Module):
    def __init__(self, envs, num_policy=6, num_node_features=50, hidden=32, history_len=7, num_actions=27, num_partition=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), actor_path="", forward_times=10):
        super().__init__()

        self.actor = PolicyEnsemble(
            num_models=num_policy, 
            num_node_features=num_node_features, 
            hidden=hidden, 
            num_actions=num_actions, 
            out_dim=num_actions, 
            std=0.01, 
            history_len=history_len, 
            num_partition=num_partition, 
            device=device
        )
        if actor_path != "":
            self.actor.load_state_dict(torch.load(actor_path))
        self.actor.train()

        self.device = device

        self.num_actions = num_actions

        self.forward_times = forward_times

        self.use_MCDropout = False

        self.epsilon = 0.05

        self.num_partition = num_partition

    def search(self, state, partition, search_type):

        # # 1. obtain logits
        # if search_type=="adaptive":
        #     means = []
        #     stds = []
        #     for base in self.actor.base_models:
        #         logits_i = []
        #         for _ in range(self.forward_times):
        #             logit = base(state, partition)
        #             logits_i.append(logit)
        #         logits_i = torch.stack(logits_i, dim=0)     # (forward_times, num_partition, num_action)

        #         mean = torch.mean(logits_i, dim=0)          # (num_partition, num_action)
        #         std = torch.std(logits_i)                   # float
            
        #         means.append(mean)
        #         stds.append(std)
        #     means = torch.stack(means, dim=0)               # (num_base, num_partition, num_action)
        #     stds = torch.tensor(stds)

        #     logits = means[torch.argmin(stds)]              # (num_partition, num_action)
        # else:

            # logits = []
            # for base in self.actor.base_models:
            #     logit = base(state, partition)
            #     logits.append(logit)
            # logits = torch.stack(logits, dim=0)             # (num_base, num_partition, num_action)

            # logits = []
            # for _ in range(3):
            #     logits.append(self.actor(state, partition))           # (num_partition, num_action)
            # logits = torch.stack(logits, dim=0)
            # logits = torch.mean(logits, dim=0)

        # 1. obtain logits
        with autocast():
            logits = self.actor(state, partition)

        # 2. obtain action
        if search_type=='random':
            # actions = []
            # for _ in range(len(self.actor.base_models)):
            #     action = torch.tensor(np.random.choice(np.arange(self.num_actions), size=20))
            #     actions.append(action)
            # actions = torch.stack(actions, dim=0)           # (num_base, num_partition)
            action = torch.tensor(np.random.choice(np.arange(self.num_actions), size=self.num_partition)).long()

        elif search_type=='epsilon-greedy':
            # actions = []
            # for i in range(len(self.actor.base_models)):
            #     rnd = torch.randn(1)
            #     if rnd<self.epsilon:
            #         action = torch.tensor(np.random.choice(np.arange(self.num_actions), size=20))
            #     else:
            #         action = torch.argmax(logits[i], dim=1)
            #     actions.append(action)
            # actions = torch.stack(actions, dim=0)           # (num_base, num_partition)
            
            rnd = torch.randn(1)
            if rnd < self.epsilon:
                action = torch.tensor(np.random.choice(np.arange(self.num_actions), size=self.num_partition)).long()
            else:
                action = torch.argmax(logits, dim=1)

        elif search_type=='sampling':
            # actions = []
            # for i in range(len(self.actor.base_models)):
            #     probs = Categorical(logits=logits[i])
            #     action = probs.sample()
            #     actions.append(action)
            # actions = torch.stack(actions, dim=0)           # (num_base, num_partition)
            probs = Categorical(logits=logits)
            action = probs.sample()

        # elif search_type=='adaptive':
        #     rnd = torch.randn(1)
        #     if rnd<self.epsilon:
        #         action = torch.tensor(np.random.choice(np.arange(self.num_actions), size=20))
        #     else:
        #         action = torch.argmax(logits, dim=1)        # (num_partition)

        # should be 2 dims: 
        # 1) whether include human feedback
        # 2) how to generate actions from policy (random/greedy/sampling)

        else:
            print("not implemented search algorithm!")
            exit()
        
        return logits, action.cuda()


def main():
    args = tyro.cli(Args)

    if args.run_name=="":
        run_name = f"{args.env_id}__{args.exp_name}__{time.strftime('%m%d%H%M%S', time.localtime())}__{args.seed}__inference"
    else:
        run_name = args.run_name

    os.makedirs(f"runs/{run_name}/", exist_ok=True)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = False

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(device)

    # data generation

    # 1. simulation
    if args.dataset == "simulation":
        data, labels = gauss_clusters(
            n_clusters=20,
            dim=50,
            pts_cluster=1000, #250, #500, #1000,
            stepsize=6,
            random_state=None,
        )
    elif args.dataset=='sc-trans':
        data = np.load("../DVHL_others/exp2/data.npy")
        labels = np.load("../DVHL_others/exp2/labels.npy")
    else:
        print("wrong dataset!")
        exit()

    if args.subset!=0:
        idx = np.random.choice(data.shape[0], args.subset, replace=False)
        data, labels = data[idx], labels[idx]

        np.save("./runs/{}/data_online_{}.npy".format(run_name, args.subset), data)
        np.save("./runs/{}/labels_online_{}.npy".format(run_name, args.subset), labels)

    num_partition = len(np.unique(labels))
    if args.dataset=='simulation':
        partition = get_partition(data, k=num_partition, labels=None).to(device)           # (data.shape[0], ) -- each entry is a cluster
    elif args.dataset=='sc-trans':
        partition = torch.from_numpy(labels).to(device)
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

    inference = True

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
        inference=inference, 
        dataset=args.dataset
    )

    agent = InferenceAgent(
        envs, 
        num_policy=6, 
        num_node_features=50, 
        hidden=32, 
        history_len=7, 
        num_actions=27, 
        num_partition=num_partition, 
        device=device, 
        actor_path=args.actor_path, 
        forward_times=10
    ).to(device)

    optimizer = optim.Adam([
        *[param for param in agent.actor.base_models[0].head.parameters()], 
        *[param for param in agent.actor.base_models[1].head.parameters()], 
        *[param for param in agent.actor.base_models[2].head.parameters()], 
        *[param for param in agent.actor.base_models[3].head.parameters()], 
        *[param for param in agent.actor.base_models[4].head.parameters()], 
        *[param for param in agent.actor.base_models[5].head.parameters()]
    ], lr=args.learning_rate, eps=1e-5)

    scaler = GradScaler()

    next_obs = envs.reset()
    actions = []
    rewards = []

    t1 = time.time()
    t2 = 0
    for i in range(args.num_steps):

        while True:
            log_prob, action = agent.search(next_obs, partition=partition, search_type=args.search)
            hp = envs.combinations[action.detach().cpu() % len(envs.combinations)]
            alpha, beta, gamma = hp[:, 0], hp[:, 1], hp[:, 2]

            alpha = {k:alpha[k] for k in range(len(alpha))}
            alpha = [alpha[i.item()] for i in partition]

            beta = {k:alpha[k] for k in range(len(beta))}
            beta = [beta[i.item()] for i in partition]

            gamma = {k:alpha[k] for k in range(len(gamma))}
            gamma = [gamma[i.item()] for i in partition]

            n_neighbors = np.round(alpha * envs.current_state["n_neighbors"]).astype(np.int32)
            MN_ratio = beta * envs.current_state["MN_ratio"]
            FP_ratio = gamma * envs.current_state["FP_ratio"]

            torch.cuda.empty_cache()
            gc.collect()

            if np.min(n_neighbors) < 1 or np.min(MN_ratio*n_neighbors) < 1 or np.min(FP_ratio*n_neighbors) < 1:
                with autocast():
                    optimizer.zero_grad()
                
                    loss = log_prob.gather(1,action.view(1,-1))         # bad action -- update

                scaler.scale(loss.mean()).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                break

        agent.train()

        print("\n\n\nstep-{}: {}".format(i, action))
        with open("./runs/{}/println.txt".format(run_name), 'a') as f:
            print("\n\n\nstep-{}: {}".format(i, action), file=f)
        

        actions.append(action.detach().cpu())

        # # next step
        # new_state, reward = transition(action, partition)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, terminations, truncations, infos = envs.next_step(action, 1, i, partition)

        rewards.append(reward.detach().cpu())

        # # for idxx in [0, 2000, 10000]:
        # for idxx in [0, 200, 500]:
        #     print("next_obs {}: ".format(idxx), next_obs["n_neighbors"][idxx], next_obs["MN_ratio"][idxx], next_obs["FP_ratio"][idxx])

        # r1 = float(input('do you think this visualisation is better than the previous one?'))
        # r2 = float(input('do you '))

        torch.save(next_obs["n_neighbors"], f"./runs/{run_name}/n_neighbours_{i}.pt")
        torch.save(next_obs["MN_ratio"], f"./runs/{run_name}/MN_ratio_{i}.pt")
        torch.save(next_obs["FP_ratio"], f"./runs/{run_name}/FP_ratio_{i}.pt")

        
        # fine-tuning policy
        if args.update:
            with autocast():
                optimizer.zero_grad()
            
                loss_1 = (-reward * log_prob).mean()        # Reinforcement learning loss
                loss_2 = -log_prob.gather(1,action.view(1,-1))         # Imitation learning loss
                loss = loss_1 + loss_2 * args.param_lambda

            scaler.scale(loss.mean()).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.mean().backward()
            # optimizer.step()

        torch.cuda.empty_cache()
        gc.collect()

    torch.save(torch.stack(actions, dim=0), f"./runs/{run_name}/actions.pt")
    torch.save(torch.tensor(rewards), f"./runs/{run_name}/rewards.pt")


if __name__=="__main__":
    main()