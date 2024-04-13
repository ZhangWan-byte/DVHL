import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.categorical import Categorical

from ppo import PolicyEnsemble
from models import SiameseNet


@dataclass
class Args:
    run_name: str = ""
    """resume previous checkpoint"""

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 3407
    """seed of the experiment"""
    num_envs: int = 1
    """number of environments"""
    actor_path: str = ""
    """path to load pre-trained actor"""

    # Algorithm specific
    env_id: str = "DVHL"
    """the id of the environment"""
    num_steps: int = 100
    """number of rollout steps"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    search: str = "sampling" # greedy / sampling / beam / eas-lay
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

    def search(self, state, partition, search_type):
        logits = self.actor(state, partition)

        if search_type=='sampling':
            probs = Categorical(logits=logits)
            action = probs.sample().item()
        
        elif search_type=='greedy':
            action = torch.argmax(logits.view(-1)).item()

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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

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
    # idx = np.random.choice(data.shape[0], 1000, replace=False)
    # data, labels = data[idx], labels[idx]

    num_partition = len(np.unique(labels))
    partition = get_partition(data, k=num_partition, labels=None).to(device)           # (data.shape[0], ) -- each entry is a cluster

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
        data.astype('float32'), 
        labels.astype('float32'), 
        model_path="./exp1/model_dropout.pt",  
        action_space=27, 
        history_len=7, 
        save_path=f"./runs/{run_name}", 
        num_steps = args.num_steps, 
        num_partition=num_partition, 
        run_name=run_name
    )
    print("data: {}, labels: {}".format(data.shape, labels.shape))

    agent = InferenceAgent(
        envs, 
        num_policy=args.num_policy, 
        num_node_features=data.shape[1], 
        hidden=32, 
        history_len=7, 
        num_actions=27, 
        num_partition=num_partition, 
        device=device, 
        actor_path=args.actor_path
    ).to(device)

    # optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    next_obs = envs.reset()
    next_done = torch.zeros(args.num_envs).to(device)

    obs = []
    actions = torch.zeros((args.num_steps, num_partition, args.num_envs)).to(device)
    logprobs = torch.zeros((args.num_steps, num_partition, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, num_partition, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)

    for i in range(args.num_steps):
        
        obs.append(next_obs)
        dones[step] = next_done

        with torch.no_grad():

            # handle accident or illegal situation
            while True:
                action, logprob, _, value = agent.get_action_and_value(next_obs, partition=partition)
                hp = envs.combinations[action.cpu() % len(envs.combinations)]
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

                if np.min(n_neighbors) < 1 or np.min(MN_ratio*n_neighbors) < 1 or np.min(FP_ratio*n_neighbors) < 1:
                    continue         
                else:
                    break



if __name__=="__main__":
    main()