from .ResNet import *
from .ScagModules import *

import numpy as np

# from metrics import scagnostics
# from metrics import ABW, CAL, DSC, HM, NH, SC, CC
# from metrics import Stress, CCA, NLM
# from metrics import LCMC, Trustworthiness, NeRV, AUClogRNX


class AttnFusion(nn.Module):
    def __init__(self, visual_size=100, metric_size=9, hidden_dim=10):
        super(AttnFusion, self).__init__()

        self.hidden_dim = hidden_dim
        # self.proj_Q = nn.Linear(metric_size, hidden_dim)
        # self.proj_K = nn.Linear(visual_size, hidden_dim)
        # self.proj_V = nn.Linear(visual_size, hidden_dim)

        self.linear1 = nn.Linear(visual_size+metric_size, visual_size+metric_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(visual_size+metric_size, hidden_dim)

    # def compute_attention(self, Q, K, V):
    #     """dot-product attention

    #     :param Q: query
    #     :param K: key
    #     :param V: value
    #     :return: weighted sum
    #     """
    #     attention_scores = F.softmax(torch.matmul(Q, K.t())/np.sqrt(self.hidden_dim), dim=-1)
    #     weighted_sum = torch.matmul(attention_scores, V)

    #     return weighted_sum

    def forward(self, visual_feature, user_preference):
        """attention and flatten

        :param user_preference: (1, 9)
        :param visual_feature: (1, 10)
        """

        # Q = self.proj_Q(user_preference)
        # K = self.proj_K(visual_feature)
        # V = self.proj_V(visual_feature)

        # # (B, 10, 10)
        # fusion_attn = self.compute_attention(Q, K, V).view(1,-1)


        vu = torch.cat([visual_feature, user_preference], dim=1)
        fusion_attn = self.linear1(vu)
        fusion_attn = self.relu1(fusion_attn)
        fusion_attn = self.linear2(fusion_attn)

        return fusion_attn


"""
    CNN + Preference Tower
"""
# class HumanModel(nn.Module):
#     def __init__(self, cnn_layers=[1,1,1,1], metric_num=9, hidden_dim=10, batch_size=1000, device=torch.device('cuda')):
#         super(HumanModel, self).__init__()
        
#         self.hidden_dim = hidden_dim
#         self.device = device

#         # cnn tower
#         self.cnn = ResNet(BasicBlock, cnn_layers)

#         # preference tower
#         self.scag_module = DAB(
#             approximator=ScagEstimator(size_in=batch_size, size_out=metric_num), 
#             hard_layer=ScagModule()
#         )
#         self.mu = nn.Parameter(torch.zeros((metric_num, 1)))
#         self.logvar = nn.Parameter(torch.log(torch.ones((metric_num, 1))))
#         self.user_weights = nn.Parameter(torch.zeros((metric_num, 1)))
#         self.pref_mlp = nn.Sequential(
#             nn.Linear(metric_num, metric_num), 
#             nn.ReLU(), 
#             nn.Linear(metric_num, metric_num), 
#             nn.ReLU(), 
#             nn.Linear(metric_num, metric_num), 
#             nn.ReLU()
#         )

#         # fusion layer
#         self.fusion = AttnFusion(visual_size=16, metric_size=metric_num, hidden_dim=hidden_dim)

#         # prediction heads
#         # Q1: 
#         self.head1 = nn.Sequential(
#             nn.Linear(self.hidden_dim, self.hidden_dim), 
#             nn.ReLU(), 
#             nn.Linear(self.hidden_dim, 5), 
#             # nn.Softmax()
#         )
#         # Q2: 
#         self.head2 = nn.Sequential(
#             nn.Linear(self.hidden_dim, self.hidden_dim), 
#             nn.ReLU(), 
#             nn.Linear(self.hidden_dim, 5), 
#             # nn.Softmax()
#         )
#         # Q3: 
#         self.head3 = nn.Sequential(
#             nn.Linear(self.hidden_dim, self.hidden_dim), 
#             nn.ReLU(), 
#             nn.Linear(self.hidden_dim, 5), 
#             # nn.Softmax()
#         )
#         # Q4: 
#         self.head4 = nn.Sequential(
#             nn.Linear(self.hidden_dim, self.hidden_dim), 
#             nn.ReLU(), 
#             nn.Linear(self.hidden_dim, 5), 
#             # nn.Softmax()
#         )

#     def calc_metrics(self, z, labels, x):
#         """calculate metric values

#         :param z: (N_low, 2) - dimensionality reduction result z
#         :param labels: (N_low, 1) - class labels for z
#         :param x: (N_high, 2) - original high dimensional data
#         """

#         # scagnostics
#         all_scags = scagnostics.compute(z[:, 0], z[:, 1])

#         # # cluster separability
#         # abw_score = ABW.compute(visu=z, labels=labels)
#         # cal_score = CAL.compute(visu=z, labels=labels)
#         # dsc_score = DSC.compute(visu=z, labels=labels)
#         # hm_score = HM.compute(visu=z, labels=labels)
#         # nh_score = NH.compute(df=pd.Dataframe(z), k=5)
#         # sc_score = SC.compute(visu=z, labels=labels)

#         # # correlation btw distances
#         # cc_score = CC.compute(data=x, visu=z)

#         # # stress
#         # nms_score = Stress.compute(data=x, visu=z)
#         # cca_score = CCA.compute(data=x, visu=z)
#         # nlm_score = NLM.compute(data=x, visu=z)

#         # # small neighbourhoods
#         # lcmc_score = LCMC.compute(dataset=x, visu=z)
#         # TC_score = Trustworthiness.compute(dataset=x, visu=z)
#         # nerv_score = NeRV.compute(data=x, visu=z, l=0.5) 

#         # # all neighbourhoods
#         # auclogrnx_score = AUClogRNX.compute(data=x, visu=z)

#         # # returned metric tensor
#         # result_tensor = torch.tensor([
#         #     list(all_scags.values()), 
#         #     bw_score, cal_score, dsc_score, hm_score, nh_score, sc_score, 
#         #     cc_score, 
#         #     nms_score, cca_score, nlm_score, 
#         #     lcmc_score, TC_score, nerv_score, 
#         #     auclogrnx_score
#         # ])

#         result_tensor = torch.tensor([list(all_scags.values())]).view(1,-1).to(self.device)

#         return result_tensor

    
#     def reparameterise(self, mu, logvar):
#         """sampling from gaussian distribution with mu and var

#         :param mu: mean of distribution
#         :param logvar: std of distribution - actually log version due to stability reason
#             see https://discuss.pytorch.org/t/vae-example-reparametrize/35843/2 for reason why log and 0.5
#         :return: a sample from distribution
#         """
        
#         if self.training:
#             std = torch.exp(0.5 * logvar)                   # logvar.mul(0.5).exp_()
#             eps = torch.randn_like(std)                     # std.data.new(std.size()).normal_()
#             return eps * std + mu                           # eps.mul(std).add_(mu)
#         else:
#             return mu

#     def forward(self, I_hat, z, labels, x):
#         """predicting human feedbacks

#         :param I_hat: approximate scatterplot of z
#         :param z: dimensionality reduction result
#         :param labels: class labels for z
#         :param x: originial high dimension data
#         :return: feedbacks
#         """

#         # cnn tower
#         visual_feature = self.cnn(I_hat)                                    # feature for visual perception

#         # preference tower
#         m = self.scag_module(z)                                             # metric values
#         d = self.reparameterise(self.mu, self.logvar).view(1,-1)            # crowd preference with random d for uncertainty

#         # maxw = torch.max(self.user_weights)
#         # minw = torch.min(self.user_weights)
#         # print("\nmaxw: {}\nminw: {}".format(maxw, minw))
#         # w = ((self.user_weights-minw)/(maxw-minw)).view(1,-1)               # personal preference over metrics
#         # print("weights: {}".format(self.user_weights))
#         # Qs_loss = NaN, maxw = minw = 0 at the beginning

#         # w = F.sigmoid(self.user_weights).view(1,-1)

#         # tanh allows different personal preference against crowd
#         w = F.tanh(self.user_weights).view(1,-1)

#         # w = F.softmax(self.user_weights / 0.1).view(1,-1)
#         # user_weights is always 0.5

#         # prod = m * F.softmax(d * w / 0.01)
#         pref_weights = d * w
#         prod = m * pref_weights
#         user_preference = self.pref_mlp(prod)                               # feature for user preference

#         # feature fusion
#         feats = self.fusion(user_preference=user_preference, visual_feature=visual_feature)

#         # prediction heads
#         a1 = self.head1(feats)
#         a2 = self.head2(feats)
#         a3 = self.head3(feats)
#         a4 = self.head3(feats)
#         # ...

#         answers = torch.vstack([a1, a2, a3, a4])

#         return answers, pref_weights, m




"""
    only CNN
"""
# class HumanModel(nn.Module):
#     def __init__(self, answers_classes=5, cnn_layers=[1,1,1,1], metric_num=9, hidden_dim=10, out_channels=[10, 16, 32, 64], batch_size=1000, device=torch.device('cuda')):
#         super(HumanModel, self).__init__()
        
#         self.hidden_dim = hidden_dim
#         self.device = device

#         self.answers_classes = answers_classes

#         # cnn tower
#         self.cnn = ResNet(block=BasicBlock, num_block=cnn_layers, num_classes=hidden_dim, out_channels=out_channels)

#         # # preference tower
#         # self.scag_module = DAB(
#         #     approximator=ScagEstimator(size_in=batch_size, size_out=metric_num), 
#         #     hard_layer=ScagModule()
#         # )
#         # self.mu = nn.Parameter(torch.zeros((metric_num, 1)))
#         # self.logvar = nn.Parameter(torch.log(torch.ones((metric_num, 1))))
#         # self.user_weights = nn.Parameter(torch.zeros((metric_num, 1)))
#         # self.pref_mlp = nn.Sequential(
#         #     nn.Linear(metric_num, metric_num), 
#         #     nn.ReLU(), 
#         #     nn.Linear(metric_num, metric_num), 
#         #     nn.ReLU(), 
#         #     nn.Linear(metric_num, metric_num), 
#         #     nn.ReLU()
#         # )

#         # # fusion layer
#         # self.fusion = AttnFusion(visual_size=16, metric_size=metric_num, hidden_dim=hidden_dim)

#         # prediction heads
#         # Q1: 
#         self.head1 = nn.Sequential(
#             nn.Linear(self.hidden_dim, self.hidden_dim), 
#             nn.ReLU(), 
#             nn.Linear(self.hidden_dim, self.answers_classes), 
#             # nn.Softmax()
#         )
#         # Q2: 
#         self.head2 = nn.Sequential(
#             nn.Linear(self.hidden_dim, self.hidden_dim), 
#             nn.ReLU(), 
#             nn.Linear(self.hidden_dim, self.answers_classes), 
#             # nn.Softmax()
#         )
#         # Q3: 
#         self.head3 = nn.Sequential(
#             nn.Linear(self.hidden_dim, self.hidden_dim), 
#             nn.ReLU(), 
#             nn.Linear(self.hidden_dim, self.answers_classes), 
#             # nn.Softmax()
#         )
#         # Q4: 
#         self.head4 = nn.Sequential(
#             nn.Linear(self.hidden_dim, self.hidden_dim), 
#             nn.ReLU(), 
#             nn.Linear(self.hidden_dim, self.answers_classes), 
#             # nn.Softmax()
#         )

#     def forward(self, I_hat, z, labels, x):
#         """predicting human feedbacks

#         :param I_hat: approximate scatterplot of z
#         :param z: dimensionality reduction result
#         :param labels: class labels for z
#         :param x: originial high dimension data
#         :return: feedbacks
#         """

#         # cnn tower
#         visual_feature = self.cnn(I_hat)                                    # feature for visual perception

#         # # preference tower
#         # m = self.scag_module(z)                                             # metric values

#         # # tanh allows different personal preference against crowd
#         # w = F.tanh(self.user_weights).view(1,-1)
#         # prod = m * w
#         # user_preference = self.pref_mlp(prod)                               # feature for user preference

#         # # feature fusion
#         # feats = self.fusion(user_preference=user_preference, visual_feature=visual_feature)

#         feats = visual_feature

#         # prediction heads
#         a1 = self.head1(feats)
#         a2 = self.head2(feats)
#         a3 = self.head3(feats)
#         a4 = self.head3(feats)
#         # ...

#         answers = torch.vstack([a1, a2, a3, a4])

#         # return answers, pref_weights, m
#         return answers


class HumanModel(nn.Module):
    def __init__(self, input_size=2, hidden=100, num_classes=10):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden)
        self.linear2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        return x