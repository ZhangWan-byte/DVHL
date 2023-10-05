from .ResNet import *

from metrics import scagnostics
from metrics import ABW, CAL, DSC, HM, NH, SC, CC
from metrics import Stress, CCA, NLM
from metrics import LCMC, Trustworthiness, NeRV, AUClogRNX

class HumanModel(nn.Module):
    def __init__(self, cnn_type='resnet18', metric_num=16):
        super(HumanModel, self).__init__()
        
        # cnn tower
        if cnn_type=="resnet18":
            self.cnn = resnet18()
        elif cnn_type=="resnet34":
            self.cnn = resnet34()
        else:
            print("HumanModel wrong cnn_type!")
            exit()

        # preference tower
        self.mu = nn.Parameter(torch.zeros((metric_num,1)))
        self.logvar = nn.Parameter(torch.log(torch.ones((metric_num,1))))
        self.user_weights = nn.Parameter(torch.rand((metric_num,1)))

        # prediction heads
        # Q1: 
        self.head1 = nn.Sequential(
            nn.Linear(100, 100), 
            nn.ReLU(), 
            nn.Linear(100, 100), 
            nn.ReLU(), 
            nn.Linear(100, 1), 
            nn.Sigmoid()
        )
        # Q2: 
        self.head2 = nn.Sequential(
            nn.Linear(100, 100), 
            nn.ReLU(), 
            nn.Linear(100, 100), 
            nn.ReLU(), 
            nn.Linear(100, 1), 
            nn.Sigmoid()
        )
        # Q3: 
        self.head3 = nn.Sequential(
            nn.Linear(100, 100), 
            nn.ReLU(), 
            nn.Linear(100, 100), 
            nn.ReLU(), 
            nn.Linear(100, 1), 
            nn.Sigmoid()
        )

    def calc_metrics(self, z, labels, x):
        """calculate metric values

        :param z: (N_low, 2) - dimensionality reduction result z
        :param labels: (N_low, 1) - class labels for z
        :param x: (N_high, 2) - original high dimensional data
        """

        # scagnostics
        all_scags = scagnostics.compute(z_umap[:, 0], z_umap[:, 1])

        # cluster separability
        abw_score = ABW.compute(visu=z, labels=labels)
        cal_score = CAL.compute(visu=z, labels=labels)
        dsc_score = DSC.compute(visu=z, labels=labels)
        hm_score = HM.compute(visu=z, labels=labels)
        nh_score = NH.compute(df=pd.Dataframe(z), k=5)
        sc_score = SC.compute(visu=z, labels=labels)

        # correlation btw distances
        cc_score = CC.compute(data=x, visu=z)

        # stress
        nms_score = Stress.compute(data=x, visu=z)
        cca_score = CCA.compute(data=x, visu=z)
        nlm_score = NLM.compute(data=x, visu=z)

        # small neighbourhoods
        lcmc_score = LCMC.compute(dataset=x, visu=z)
        TC_score = Trustworthiness.compute(dataset=x, visu=z)
        nerv_score = NeRV.compute(data=x, visu=z, l=0.5) 

        # all neighbourhoods
        auclogrnx_score = AUClogRNX.compute(data=x, visu=z)

        # returned metric tensor
        result_tensor = torch.tensor([
            list(all_scags.values()), 
            bw_score, cal_score, dsc_score, hm_score, nh_score, sc_score, 
            cc_score, 
            nms_score, cca_score, nlm_score, 
            lcmc_score, TC_score, nerv_score, 
            auclogrnx_score
        ])

        return result_tensor

    
    def reparameterise(self, mu, logvar):
        """sampling from gaussian distribution with mu and var

        :param mu: mean of distribution
        :param logvar: std of distribution - actually log version due to stability reason
            see https://discuss.pytorch.org/t/vae-example-reparametrize/35843/2 for reason why log and 0.5
        :return: a sample from distribution
        """
        
        if self.training:
            std = torch.exp(0.5 * logvar) # logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) # std.data.new(std.size()).normal_()
            return eps * std + mu # eps.mul(std).add_(mu)
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
        visual_feature = self.cnn(I_hat)

        # preference tower
        m = self.calc_metric(z=z, labels=labels, x=x)       # metric values
        d = self.reparameterise(self.mu, self.logvar)       # random d for uncertainty
        w = F.sigmoid(self.user_weights)                    # quasi-binary w for personal preference over metrics
        user_preference = m * d * w

        # prediction heads
        q1 = self.head1(visual_feature, user_preference)
        q2 = self.head2(visual_feature, user_preference)
        q3 = self.head3(visual_feature, user_preference)
        # ...

        return q1, q2, q3