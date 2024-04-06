# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
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

from annoy import AnnoyIndex
from myPaCMAP import distance_to_option
from env import DREnv

import warnings
warnings.filterwarnings('ignore')

import pickle
from copy import deepcopy
from tqdm import tqdm

import torchvision as tv
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.utils import save_image

@dataclass
class Args:
    run_name: str = ""
    """resume previous checkpoint"""

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
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

    # Algorithm specific arguments
    env_id: str = "DVHL" # "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 120 # 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1 # 4
    """the number of parallel game environments"""
    num_steps: int = 32 # 6 # 10 # 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True # False # True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 2 # 4
    """the number of mini-batches"""
    update_epochs: int = 4
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

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def get_scaledKNN(X, _RANDOM_STATE=None):
    n, dim = X.shape
    # sample more neighbors than needed
    n_neighbors_extra = min(n_neighbors + 50, n - 1)
    tree = AnnoyIndex(dim, metric=distance)
    if _RANDOM_STATE is not None:
        tree.set_seed(_RANDOM_STATE)
    for i in range(n):
        tree.add_item(i, X[i, :])
    tree.build(20)

    option = distance_to_option(distance=distance)

    nbrs = np.zeros((n, n_neighbors_extra), dtype=np.int32)
    knn_distances = np.empty((n, n_neighbors_extra), dtype=np.float32)

    for i in range(n):
        nbrs_ = tree.get_nns_by_item(i, n_neighbors_extra + 1)
        nbrs[i, :] = nbrs_[1:]
        for j in range(n_neighbors_extra):
            knn_distances[i, j] = tree.get_distance(i, nbrs[i, j])
    print_verbose("Found nearest neighbor", verbose)
    sig = np.maximum(np.mean(knn_distances[:, 3:6], axis=1), 1e-10)
    print_verbose("Calculated sigma", verbose)
    scaled_dist = scale_dist(knn_distances, sig, nbrs)

    return scaled_dist

class GAT(torch.nn.Module):
    def __init__(self, num_node_features=50, hidden=32, num_actions=81, out_dim=1, std=1.0, device=torch.device('cpu')):
        super().__init__()
        self.hidden = hidden
        self.num_actions = num_actions
        self.device = device
        self.edge_dim = 4
        self.out_dim = out_dim

        self.conv1 = GATv2Conv(num_node_features, hidden, edge_dim=self.edge_dim)
        self.conv2 = GATv2Conv(hidden, hidden, edge_dim=self.edge_dim)
        
        # graph feature
        self.pooling = MultiheadAttention(embed_dim=hidden, num_heads=2)        
        self.mlp = nn.Sequential(
            layer_init(nn.Linear(in_features=hidden, out_features=hidden)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(in_features=hidden, out_features=hidden)),
            nn.LeakyReLU()
        )

        # history feature
        self.gru = nn.GRU(input_size=num_actions+2, hidden_size=hidden, num_layers=1)

        # prediction head
        self.head = nn.Sequential(
            layer_init(nn.Linear(in_features=hidden*(1+self.edge_dim), out_features=hidden)), 
            nn.LeakyReLU(),
            layer_init(nn.Linear(in_features=hidden, out_features=self.out_dim), std=std)
        )

    def forward(self, state):

        x, edge_index, edge_attr, history = state["x"].float(), state["edge_index"].long(), state["edge_attr"].float(), state["history"].float()

        n, f = x.shape[0], self.hidden

        # Conv layers
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)

        # Readout layer
        x, _ = self.pooling(x, x, x)
        # x = x.sum(dim=0)
        x, _ = torch.max(x, dim=0)
        x = self.mlp(x.view(1,-1)).view(1,-1)

        # History features
        out, _ = self.gru(history)
        # out = out.max(dim=0).view(1,-1)
        out = out.flatten().view(1, -1)

        # Prediction head
        out = torch.cat([x, out], dim=1)
        out = self.head(out)

        return out



class Agent(nn.Module):
    def __init__(self, envs, num_node_features=50, hidden=64, num_actions=81):
        super().__init__()
        # self.critic = nn.Sequential(
        #     layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, 1), std=1.0),
        # )
        # self.actor = nn.Sequential(
        #     layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        # )

        self.critic = GAT(num_node_features, hidden, num_actions=num_actions, out_dim=1, std=1.0)
        self.actor = GAT(num_node_features, hidden, num_actions=num_actions, out_dim=num_actions, std=0.01)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_value(self, state):
        # for key, value in state.items():
        #     state[key] = state[key].to(self.device)

        return self.critic(state)

    def get_action_and_value(self, state, action=None):
        # for key, value in state.items():
        #     try:
        #         state[key] = state[key].to(self.device)
        #     except:
        #         pass

        if type(state)==type([]):
            # print(len(state))
            ac = []
            pbs = []
            ent = []
            value = []

            for i in range(len(state)):
                logits = self.actor(state[i])
                probs = Categorical(logits=logits)
                if action is None:
                    action = probs.sample()
                
                ac.append(action)
                pbs.append(probs.log_prob(action))
                ent.append(probs.entropy())
                value.append(self.critic(state[i]))
            ac = torch.vstack(ac)
            pbs = torch.vstack(pbs)
            ent = torch.vstack(ent)
            value = torch.vstack(value)
            # print(ac, pbs, ent, value)
            return ac, pbs, ent, value
        else:
            logits = self.actor(state)
            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            
            return action, probs.log_prob(action), probs.entropy(), self.critic(state)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)                   # 12
    args.minibatch_size = int(args.batch_size // args.num_minibatches)      # 4
    args.num_iterations = args.total_timesteps // args.batch_size           # [120/12]=10
    
    if args.run_name=="":
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{time.strftime('%m%d%H%M%S', time.localtime())}"
    else:
        run_name = args.run_name
    # run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{time.strftime('%m%d%H%M%S', time.localtime())}"

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

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    # )
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    
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
    data, labels = data[idx], labels[idx]

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
    envs = DREnv(
        data.astype('float32'), 
        labels.astype('float32'), 
        model_path="./exp1/model_dropout.pt", 
        batch_size=1000, 
        action_space=81, 
        history_len=3, 
        save_path=f"runs/{run_name}", 
        num_steps = args.num_steps
    )

    agent = Agent(envs, num_node_features=data.shape[1], hidden=16, num_actions=81).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    print("actor params: {}".format(sum([p.numel() for p in agent.actor.parameters()])))
    print("critic params: {}".format(sum([p.numel() for p in agent.critic.parameters()])))

    # ALGO Logic: Storage setup
    # obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    # actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    # logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()

    all_rewards = []
    all_actions = []

    for iteration in range(1, args.num_iterations + 1):
        # reset obs each iteration, as other variables do. otherwise OOM.
        
        # next_obs, _ = envs.reset(seed=args.seed)
        next_obs = envs.reset()
        # next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs).to(device)

        obs = []
        actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            # obs[step] = next_obs
            obs.append(next_obs)
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                # try:

                # handle accident or illegal situation
                while True:
                    action, logprob, _, value = agent.get_action_and_value(next_obs)

                    alpha, beta, gamma, hetero_homo = envs.combinations[action % len(envs.combinations)]
                    n_neighbors = int(alpha * envs.current_state["n_neighbors"])
                    MN_ratio = beta * envs.current_state["MN_ratio"]
                    FP_ratio = gamma * envs.current_state["FP_ratio"]

                    if n_neighbors <= 0 or MN_ratio*n_neighbors < 1 or FP_ratio*n_neighbors < 1:
                        continue
                    else:
                        break

                # except:
                #     print("error happened! ", action, logprob, value)
                #     pickle.dump(next_obs, open("./error_next_obs.pkl", "wb"))
                #     torch.save(torch.tensor(envs.history_rewards), "./error_history_rewards1.pt")
                #     torch.save(torch.tensor(envs.history_actions), "./error_history_actions1.pt")
                #     torch.save(agent.actor.state_dict(), "./error_actor1.pt")
                #     torch.save(agent.critic.state_dict(), "./error_critic1.pt")
                #     exit()

                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.next_step(action.cpu().numpy().item(), iteration, step)
            # next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            # next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            next_done = torch.tensor(next_done).to(device)

            print("\nhistory_len:{}\nhistory_rewards:{}\nhistory_actions:{}\nreward:{}\nnext_done:{}".format(
                envs.history_len, envs.history_rewards, envs.history_actions, reward, 
                next_done.detach().cpu().item()))
            print("best: {}".format(envs.best_name))

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            all_rewards.append(reward)
            all_actions.append(envs.history_actions[-1])

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
            print("advantages, returns: ", advantages, returns)

        # flatten the batch
        # b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        # b_logprobs = logprobs.reshape(-1)
        # b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        # b_advantages = advantages.reshape(-1)
        # b_returns = returns.reshape(-1)
        # b_values = values.reshape(-1)

        b_obs = deepcopy(obs)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        # print(actions.shape, b_actions.shape)
        # print(actions, b_actions)
        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in tqdm(range(0, args.batch_size, args.minibatch_size)):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                # print(mb_inds, b_actions)
                # bobs = b_obs[mb_inds]
                # bacs = b_actions.long()[mb_inds]
                # print(bobs, bacs)
                # _, newlogprob, entropy, newvalue = agent.get_action_and_value(bobs, bacs)
                # print(len(b_obs), mb_inds)
                b_obs_i = [b_obs[i] for i in mb_inds]
                b_actions_i = b_actions.long()[mb_inds]
                # print(b_obs_i, b_actions_i)
                # print(mb_inds)
                try:
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs_i, b_actions_i)
                except:
                    print(newlogprob, entropy, newvalue)
                    pickle.dump(b_obs_i, open("./error_b_obs_i.pkl", "wb"))
                    torch.save(torch.tensor(envs.history_rewards), "./error_history_reward2.pt")
                    torch.save(torch.tensor(envs.history_actions), "./error_history_actions2.pt")
                    torch.save(agent.actor.state_dict(), "./error_actor2.pt")
                    torch.save(agent.critic.state_dict(), "./error_critic2.pt")
                    exit()

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                print("logratio: ", logratio)
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                print("mb_advantages: ", mb_advantages)
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                print("mb_advantages, ratio: ", mb_advantages, ratio)
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                print("loss: ", loss, pg_loss, entropy_loss, v_loss)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

    torch.save(agent.state_dict(), "runs/{}/agent.pt".format(run_name))
    torch.save(torch.tensor(all_rewards), "runs/{}/all_rewards.pt".format(run_name))
    torch.save(torch.tensor(all_actions), "runs/{}/all_actions.pt".format(run_name))