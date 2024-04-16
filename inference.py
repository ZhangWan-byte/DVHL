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


@dataclass
class Args:
    run_name: str = ""
    """resume previous checkpoint"""

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
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
    num_steps: int = 100
    """number of rollout steps"""
    num_policy: int = 6
    """number of policies"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    search: str = "sampling" # random / greedy / sampling / beam / eas-lay
    """types of search algorithm"""


class InferenceAgent(nn.Module):
    def __init__(self, envs, num_policy=6, num_node_features=50, hidden=64, history_len=7, num_actions=27, num_partition=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), actor_path=None):
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
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))

        self.device = device

        self.num_actions = num_actions

    def search(self, state, partition, search_type):
        logits = self.actor(state, partition)

        if search_type=='random':
            action = torch.tensor(np.random.choice(np.arange(self.num_actions), size=20)).view(-1).to(self.device)
            
        elif search_type=='sampling':
            probs = Categorical(logits=logits)
            action = probs.sample()
        
        elif search_type=='greedy':
            action = torch.argmax(logits, dim=1)

        elif search_type=='beam':
            pass

        elif search_type=='eas-lay':
            pass

        else:
            print("not implemented search algorithm!")
            exit()
        
        return action


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

    # data generation

    # 1. simulation
    data, labels = gauss_clusters(
        n_clusters=20,
        dim=50,
        pts_cluster=1000,
        stepsize=6,
        random_state=None,
    )
    idx = np.random.choice(data.shape[0], 1000, replace=False)
    data_1k, labels_1k = data[idx], labels[idx]

    num_partition = len(np.unique(labels))
    partition = get_partition(data, k=num_partition, labels=None).to(device)           # (data.shape[0], ) -- each entry is a cluster
    partition_1k = get_partition(data_1k, k=num_partition, labels=None).to(device)
    print("partition: {}, partition_1k: {}".format(partition.shape, partition_1k.shape))

    # 2. MNIST
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # trainset = tv.datasets.MNIST(root='./data',  train=True, download=False, transform=transform)
    # traindata = [i[0].unsqueeze(0) for i in trainset]
    # trainlabel = [i[1] for i in trainset]
    # data = torch.vstack(traindata).numpy().reshape(60000, 28*28)
    # labels = torch.tensor(trainlabel).numpy().reshape(60000, 1)
    # idx = np.random.choice(data.shape[0], 10000, replace=False)
    # data, labels = data[idx], labels[idx]

    envs = DREnv(
        data_1k.astype('float32'), 
        labels_1k.astype('float32'), 
        model_path="./exp1/model_dropout.pt",  
        action_space=27, 
        history_len=7, 
        save_path=f"./runs/{run_name}", 
        num_steps = args.num_steps, 
        num_partition=num_partition, 
        run_name=run_name, 
        inference=True, 
        data=data, 
        labels=labels, 
        idx=idx
    )
    print("data_1k: {}, labels_1k: {}".format(data_1k.shape, labels_1k.shape))

    agent = InferenceAgent(
        envs, 
        num_policy=args.num_policy, 
        num_node_features=data_1k.shape[1], 
        hidden=32, 
        history_len=7, 
        num_actions=27, 
        num_partition=num_partition, 
        device=device, 
        actor_path=args.actor_path
    ).to(device)

    # optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    next_obs = envs.reset()
    actions = []
    rewards = []

    t1 = time.time()
    t2 = 0
    for i in range(args.num_steps):
        
        with torch.no_grad():

            # handle accident or illegal situation
            while True:
                action = agent.search(next_obs, partition=partition_1k, search_type=args.search)
                print("\n\n\nstep-{}: {}".format(i, action.detach().cpu()))
                hp = envs.combinations[action.cpu() % len(envs.combinations)]
                alpha, beta, gamma = hp[:, 0], hp[:, 1], hp[:, 2]

                alpha = {k:alpha[k] for k in range(len(alpha))}
                alpha = [alpha[i.item()] for i in partition_1k]

                beta = {k:beta[k] for k in range(len(beta))}
                beta = [beta[i.item()] for i in partition_1k]

                gamma = {k:gamma[k] for k in range(len(gamma))}
                gamma = [gamma[i.item()] for i in partition_1k]

                n_neighbors = np.round(alpha * envs.current_state["n_neighbors"]).astype(np.int32)
                MN_ratio = beta * envs.current_state["MN_ratio"]
                FP_ratio = gamma * envs.current_state["FP_ratio"]

                if np.min(n_neighbors) < 1 or np.min(MN_ratio*n_neighbors) < 1 or np.min(FP_ratio*n_neighbors) < 1:
                    continue         
                else:
                    break

                t2 = time.time()
                if (t2-t1)>600:
                    break
        
        if (t2-t1)>600:
            break

        torch.cuda.empty_cache()
        gc.collect()

        actions.append(action.detach().cpu())

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, terminations, truncations, infos = envs.next_step(action.detach().cpu(), 1, i, partition_1k)

        rewards.append(reward)

        # # for idxx in [0, 2000, 10000]:
        # for idxx in [0, 200, 500]:
        #     print("next_obs {}: ".format(idxx), next_obs["n_neighbors"][idxx], next_obs["MN_ratio"][idxx], next_obs["FP_ratio"][idxx])

        # r1 = float(input('do you think this visualisation is better than the previous one?'))
        # r2 = float(input('do you '))

        torch.save(next_obs["n_neighbors"], f"./runs/{run_name}/n_neighbours_{i}.pt")
        torch.save(next_obs["MN_ratio"], f"./runs/{run_name}/MN_ratio_{i}.pt")
        torch.save(next_obs["FP_ratio"], f"./runs/{run_name}/FP_ratio_{i}.pt")


    torch.save(torch.stack(actions, dim=0), f"./runs/{run_name}/actions.pt")
    torch.save(torch.tensor(reward), f"./runs/{run_name}/rewards.pt")


if __name__=="__main__":
    main()