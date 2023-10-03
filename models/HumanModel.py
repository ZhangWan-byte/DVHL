from .ResNet import *

class HumanModel(nn.Module):
    def __init__(self, cnn_type='resnet18'):
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
        self.mu = nn.Parameter(torch.zeros((16,1)))
        self.logvar = nn.Parameter(torch.ones((16,1)))
        self.user_weights = nn.Parameter(torch.rand((16,1)))
    
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

    def forward(self, x):

        # cnn tower
        visual_feature = self.cnn(x)

        # preference tower
        m = self.calc_metric(x)
        d = self.reparameterise(self.mu, self.logvar)
        w = F.sigmoid(self.user_weights)
        user_preference = m * d * w

        # prediction heads
        q1 = self.q1(visual_feature, user_preference)
        q2 = self.q1(visual_feature, user_preference)
        q3 = self.q1(visual_feature, user_preference)
        # ...

        return q1, q2, q3