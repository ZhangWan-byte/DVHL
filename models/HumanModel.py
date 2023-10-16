from .ResNet import *

from metrics import scagnostics
from metrics import ABW, CAL, DSC, HM, NH, SC, CC
from metrics import Stress, CCA, NLM
from metrics import LCMC, Trustworthiness, NeRV, AUClogRNX


class AttnFusion(nn.Module):
    def __init__(self, visual_size=100, metric_size=9, hidden_dim=100):
        super(AttnFusion, self).__init__()

        self.hidden_dim = hidden_dim
        self.proj_Q = nn.Linear(metric_size, hidden_dim)
        self.proj_K = nn.Linear(visual_size, hidden_dim)
        self.proj_V = nn.Linear(visual_size, hidden_dim)

    def compute_attention(self, Q, K, V):
        """dot-product attention

        :param Q: query
        :param K: key
        :param V: value
        :return: weighted sum
        """
        attention_scores = F.softmax(torch.matmul(Q, K.t()), dim=-1)
        weighted_sum = torch.matmul(attention_scores, V)

        return weighted_sum

    def forward(self, visual_feature, user_preference):
        """attention and flatten

        :param user_preference: (1, 9)
        :param visual_feature: (1, 100)
        """

        Q = self.proj_Q(user_preference)
        K = self.proj_K(visual_feature)
        V = self.proj_V(visual_feature)

        # (B, 100, 100)
        fusion_attn = self.compute_attention(Q, K, V).view(1,-1)

        return fusion_attn



class HumanModel(nn.Module):
    def __init__(self, cnn_layers=[1,1,1,1], metric_num=9, hidden_dim=100, device=torch.device('cuda')):
        super(HumanModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.device = device

        # cnn tower
        self.cnn = ResNet(BasicBlock, cnn_layers)

        # preference tower
        self.mu = nn.Parameter(torch.zeros((metric_num,1)))
        self.logvar = nn.Parameter(torch.log(torch.ones((metric_num,1))))
        self.user_weights = nn.Parameter(torch.rand((metric_num,1)))
        self.pref_mlp = nn.Sequential(
            nn.Linear(metric_num, metric_num), 
            nn.ReLU(), 
            nn.Linear(metric_num, metric_num), 
            nn.ReLU()
        )

        # fusion layer
        self.fusion = AttnFusion(visual_size=100, metric_size=metric_num, hidden_dim=hidden_dim)

        # prediction heads
        # Q1: 
        self.head1 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), 
            nn.ReLU(), 
            nn.Linear(self.hidden_dim, 1), 
            nn.Sigmoid()
        )
        # Q2: 
        self.head2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), 
            nn.ReLU(), 
            nn.Linear(self.hidden_dim, 1), 
            nn.Sigmoid()
        )
        # Q3: 
        self.head3 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), 
            nn.ReLU(), 
            nn.Linear(self.hidden_dim, 1), 
            nn.Sigmoid()
        )

    def calc_metrics(self, z, labels, x):
        """calculate metric values

        :param z: (N_low, 2) - dimensionality reduction result z
        :param labels: (N_low, 1) - class labels for z
        :param x: (N_high, 2) - original high dimensional data
        """

        # scagnostics
        all_scags = scagnostics.compute(z[:, 0], z[:, 1])

        # # cluster separability
        # abw_score = ABW.compute(visu=z, labels=labels)
        # cal_score = CAL.compute(visu=z, labels=labels)
        # dsc_score = DSC.compute(visu=z, labels=labels)
        # hm_score = HM.compute(visu=z, labels=labels)
        # nh_score = NH.compute(df=pd.Dataframe(z), k=5)
        # sc_score = SC.compute(visu=z, labels=labels)

        # # correlation btw distances
        # cc_score = CC.compute(data=x, visu=z)

        # # stress
        # nms_score = Stress.compute(data=x, visu=z)
        # cca_score = CCA.compute(data=x, visu=z)
        # nlm_score = NLM.compute(data=x, visu=z)

        # # small neighbourhoods
        # lcmc_score = LCMC.compute(dataset=x, visu=z)
        # TC_score = Trustworthiness.compute(dataset=x, visu=z)
        # nerv_score = NeRV.compute(data=x, visu=z, l=0.5) 

        # # all neighbourhoods
        # auclogrnx_score = AUClogRNX.compute(data=x, visu=z)

        # # returned metric tensor
        # result_tensor = torch.tensor([
        #     list(all_scags.values()), 
        #     bw_score, cal_score, dsc_score, hm_score, nh_score, sc_score, 
        #     cc_score, 
        #     nms_score, cca_score, nlm_score, 
        #     lcmc_score, TC_score, nerv_score, 
        #     auclogrnx_score
        # ])

        result_tensor = torch.tensor([list(all_scags.values())]).view(1,-1).to(self.device)

        return result_tensor

    
    def reparameterise(self, mu, logvar):
        """sampling from gaussian distribution with mu and var

        :param mu: mean of distribution
        :param logvar: std of distribution - actually log version due to stability reason
            see https://discuss.pytorch.org/t/vae-example-reparametrize/35843/2 for reason why log and 0.5
        :return: a sample from distribution
        """
        
        if self.training:
            std = torch.exp(0.5 * logvar)                   # logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)                     # std.data.new(std.size()).normal_()
            return eps * std + mu                           # eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, I_hat, z, labels, x):
        """predicting human feedbacks

        :param I_hat: approximate scatterplot of z
        :param z: dimensionality reduction result
        :param labels: class labels for z
        :param x: originial high dimension data
        :return: feedbacks
        """

        # cnn tower
        visual_feature = self.cnn(I_hat)                                    # feature for visual perception

        # preference tower
        m = self.calc_metrics(z=z, labels=labels, x=x).view(1,-1)           # metric values
        d = self.reparameterise(self.mu, self.logvar).view(1,-1)            # random d for uncertainty
        w = F.sigmoid(self.user_weights).view(1,-1)                         # quasi-binary w for personal preference over metrics
        prod = m * d * w
        user_preference = self.pref_mlp(prod)                               # feature for user preference

        # feature fusion
        feats = self.fusion(user_preference=user_preference, visual_feature=visual_feature)

        # prediction heads
        q1 = self.head1(feats)
        q2 = self.head2(feats)
        q3 = self.head3(feats)
        # ...

        return q1, q2, q3