import torch
import torch.nn as nn


class SIMAMAttention(torch.nn.Module):
    def __init__(self, cfg):
        super(SIMAMAttention, self).__init__()
        self.cfg = cfg
        assert self.cfg.channelattention.name == "simam"
        self.activaton = nn.Sigmoid()
        self.e_lambda = 1e-4

    def forward(self, x):

        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return self.activaton(y)
