
from multiprocessing.sharedctypes import Value
import einops

import torch
import hydra
from torch import DictType, nn
from .utils import append_dims
from omegaconf import DictConfig

from .mlps import *


class GCTimeScoreNetwork(nn.Module):

    def __init__(
            self,
            cond_dim: int,
            input_dim: int,
            hidden_dim: int,
            embed_fn: DictConfig,
            n_layer: int = 2,
            output_dim=1,
            dropout: int = 0,
            activation: str = "ReLU",
            model_style: str = True,
            use_norm: bool = False,
            norm_style: str = 'BatchNorm',
            use_spectral_norm: bool = False,
            device: str = 'cuda',
    ):
        super(GCTimeScoreNetwork, self).__init__()
        self.network_type = "mlp"
        #  Gaussian random feature embedding layer for time
        self.embed = hydra.utils.call(embed_fn)
        self.time_embed_dim = embed_fn.time_embed_dim
        input_dim = self.time_embed_dim + cond_dim + input_dim 
        # set up the network
        if model_style:
            self.layers = ResidualMLPNetwork(
                input_dim,
                hidden_dim,
                n_layer,
                output_dim,
                dropout,
                activation,
                use_spectral_norm,
                use_norm,
                norm_style,
                device
            ).to(device)
        else:
            self.layers = MLPNetwork(
                input_dim,
                hidden_dim,
                n_layer,
                output_dim,
                dropout,
                activation,
                use_spectral_norm,
                device
            ).to(device)

        # build the activation layer
        self.act = return_activiation_fcn(activation)
        self.device = device
        self.training = True

    def forward(self, action, sigma, cond):
        
        embed = self.embed(sigma)
        # embed = self.embed(t)
        if len(cond.shape) == 3:
            embed = einops.rearrange(embed, 'b d -> b 1 d')
        x = torch.cat([cond, action, embed], dim=-1) 
        x = self.layers(x) 
        return x  # / marginal_prob_std(t, self.sigma, self.device)[:, None]

    def mask_cond(self, cond, force_mask=False):
        """
        Only needed for Classifier-Free Guidance (CFG)
        """
        bs, t, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones((bs, t, d), device=cond.device) * self.cond_mask_prob)# .view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def get_params(self):
        return self.parameters()
