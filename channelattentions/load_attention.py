from .senet import SEAttention
from .depthwise import DepthWiseAttention
from .cbam import CBAMAttention
from .simam import SIMAMAttention
from .depthwise_1d import DepthWise1DAttention


def load_channel_attention_module(cfg):
    if cfg.channelattention.name == "senet":
        print("Loading SE Attention")
        return SEAttention(cfg)
    elif cfg.channelattention.name == "cbam":
        print("Loading CBAM Attention")
        return CBAMAttention(cfg)
    elif cfg.channelattention.name == "simam":
        print("Loading SIMAM Attention")
        return SIMAMAttention(cfg)
    elif cfg.channelattention.name == "depthwise":
        print("Loading DepthWise Attention")
        return DepthWiseAttention(cfg)
    elif cfg.channelattention.name == "depthwise_1d":
        return DepthWise1DAttention(cfg)
    else:
        raise NotImplementedError
