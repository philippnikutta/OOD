import torch
from torch.nn import functional as F


def loss_function(recon_x, x, mu, logvar):
    # TODO only accepts inputs within [0,1]
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 4224), reduction='sum')
    L1 = F.l1_loss(recon_x, x.view(-1, 4224), reduction='mean')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return L1 + KLD, [L1, KLD]
