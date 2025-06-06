import torch
import torch.nn.functional as F
import numpy as np
import math

from torch import nn


class nETM(nn.Module):
    def __init__(self, num_topics, vocab_size, t_hidden_size, rho_size, emb_size,
                 theta_act, embeddings=None, train_embeddings=True, enc_drop=0.5,
                 topic_perturb=1):
        super(nETM, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ## define hyperparameters
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.t_hidden_size = t_hidden_size
        self.rho_size = rho_size
        self.enc_drop = enc_drop
        self.emb_size = emb_size
        self.t_drop = nn.Dropout(enc_drop)

        self.EPSILON = 1e-5
        self.neg_method = 2
        self.topic_perturb=topic_perturb
        print("neg_method:{} :->Training Using Negative Sampling".format(self.neg_method))
        print("topic_perturb=", self.topic_perturb)
        self.margin = 1
        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=2)

        self.theta_act = self.get_activation(theta_act)

        ## define the word embedding matrix \rho
        if train_embeddings:
            self.rho = nn.Linear(rho_size, vocab_size, bias=False)
        else:
            num_embeddings, emb_size = embeddings.size()
            rho = nn.Embedding(num_embeddings, emb_size)
            self.rho = embeddings.clone().float().to(self.device)

        ## define the matrix containing the topic embeddings
        self.alphas = nn.Linear(rho_size, num_topics, bias=False)#nn.Parameter(torch.randn(rho_size, num_topics))

        ## define variational distribution for \theta_{1:D} via amortizartion
        self.q_theta = nn.Sequential(
                nn.Linear(vocab_size, t_hidden_size),  self.theta_act,
                nn.Linear(t_hidden_size, t_hidden_size), self.theta_act,
            )
        self.mu_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)

    def get_activation(self, act):
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'softplus':
            act = nn.Softplus()
        elif act == 'sigmoid':
            act = nn.Sigmoid()
        elif act == 'rrelu':
            act = nn.RReLU()
        elif act == 'leakyrelu':
            act = nn.LeakyReLU()
        elif act == 'elu':
            act = nn.ELU()
        elif act == 'selu':
            act = nn.SELU()
        elif act == 'glu':
            act = nn.GLU() #error using glu
        else:
            print('Defaulting to tanh activations...')
            act = nn.Tanh()
        return act

    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def encode(self, bows):
        """Returns paramters of the variational distribution for \theta.

        input: bows
                batch of bag-of-words...tensor of shape bsz x V
        output: mu_theta, log_sigma_theta
        """
        q_theta = self.q_theta(bows)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        kl_theta = -0.5 * torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean()
        return mu_theta, logsigma_theta, kl_theta

    def get_beta(self):
        try:
            logit = self.alphas(self.rho.weight) # torch.mm(self.rho, self.alphas)
        except:
            logit = self.alphas(self.rho)
        beta = F.softmax(logit, dim=0).transpose(1, 0) ## softmax over vocab dimension
        return beta

    def get_theta(self, normalized_bows):
        mu_theta, logsigma_theta, kld_theta = self.encode(normalized_bows)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1)
        return theta, kld_theta

    def decode(self, theta, beta, theta_neg=None):
        word_dist = torch.mm(theta, beta)
        if (self.neg_method == 2):
            word_dist_neg = torch.mm(theta_neg, beta)
            return word_dist, word_dist_neg
        else:
            return word_dist

    def forward(self, bows, normalized_bows, theta=None, aggregate=True):
        ## get \theta
        if theta is None:
            theta, kld_theta = self.get_theta(normalized_bows)
        else:
            kld_theta = None

        ## get \beta
        beta = self.get_beta()

        ## get prediction loss
        if (self.neg_method == 2):
            theta_neg = nETM.perturbTheta(theta, self.topic_perturb)
            word_dist, word_dist_neg = self.decode(theta, beta, theta_neg)
            TL = self.triplet_loss(word_dist, bows, word_dist_neg)
            if aggregate:
                TL = TL.mean()
            else:
                TL = TL.sum(1)
        else:
            word_dist = self.decode(theta, beta)
            TL = 0.0
        
        preds = torch.log(word_dist+1e-6)

        recon_loss = -(preds * bows).sum(1)
        if aggregate:
            recon_loss = recon_loss.mean()
        return recon_loss, kld_theta, TL
    
    @staticmethod
    def perturb(x):
        """Add Gaussian noise."""
        #print("perturb called")
        eps = torch.randn_like(x)
        return eps.add_(x)

    @staticmethod
    def perturbTopK(x, k):
        """Add Gaussian noise."""
        _, kidx = x.topk(k=k, dim=1)
        y = x.clone()
        y[torch.arange(y.size(0))[:, None], kidx] = 0.0
        return y

    @staticmethod
    def perturbTheta(x, k):
        x_new =  nETM.perturbTopK(x, k)
        x_new = x_new/x_new.sum(dim=-1).unsqueeze(1) 
        return x_new

