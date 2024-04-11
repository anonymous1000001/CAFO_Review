from torch import nn

"""https://github.com/ZjjConan/SimAM/blob/master/networks/attentions/se_module.py"""


class SEAttention(nn.Module):
    def __init__(self, cfg):
        super(SEAttention, self).__init__()
        self.cfg = cfg
        assert self.cfg.channelattention.name == "senet"
        in_planes = cfg.task.in_channels
        hid_planes = cfg.channelattention.reduction_size

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, hid_planes, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hid_planes, in_planes, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        channel_attention = self.fc(y).view(b, c, 1, 1)
        return channel_attention
