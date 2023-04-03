from multiprocessing.sharedctypes import Value

import copy 
import hydra
from torch import DictType, nn
from .utils import append_dims
import torch 
import torch.nn.functional as F
from einops import rearrange, reduce
'''
Wrappers for the score-based models based on Karras et al. 2022
They are used to get improved scaling of different noise levels, which
improves training stability and model performance 

Code is adapted from:

https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/layers.py
'''


class GCDenoiser(nn.Module):
    """A Karras et al. preconditioner for denoising diffusion models."""

    def __init__(self, inner_model, sigma_data=1.):
        super().__init__()
        self.inner_model = inner_model
        self.sigma_data = sigma_data

    def get_scalings(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def loss(self, noised_action, action, cond, sigma):
        """
        Method to compute the loss for the denoiser during training adapted 
        for the Transformer model from Chi et al. 2023
        
        :param obs: obs of the environment
        :param action: action taken by the expert
        :param goal: goal of the agent
        :param noise: noise added to the action
        :param sigma: sampled noise levels
        """
        noised_input = noised_action
            
        c_skip, c_out, c_in = [append_dims(x, action.ndim) for x in self.get_scalings(sigma)]
        sigma = sigma.log() / 4
        # Compute the model output 
        pred = self.inner_model(noised_input * c_in, sigma, cond)
        # Compute the target for the denoiser using the scaling factors
        target = (action - c_skip * noised_input) / c_out
        loss = F.mse_loss(pred, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss

    def forward(self, action, sigma, cond, ):
        """
        Method to compute the denoised action during rollout for the agent
        """
        c_skip, c_out, c_in = [append_dims(x, action.ndim) for x in self.get_scalings(sigma)]
        sigma = sigma.log() / 4
        return self.inner_model(action * c_in, sigma, cond) * c_out + action * c_skip

    def get_params(self):
        return self.inner_model.parameters()
    
    def parameters(self, recurse=True):
        for name, param in self.inner_model.named_parameters(recurse=recurse):
            yield param
    
    
class GCDenoiserDiffusionTransformer(nn.Module):
    """A Karras et al. preconditioner for denoising diffusion models adapted for the 
    proposed Transformer model from Chi et al. 2023."""

    def __init__(self, inner_model, sigma_data=1.):
        super().__init__()
        self.inner_model = inner_model
        self.sigma_data = sigma_data

    def get_scalings(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def loss(self, noised_action, action, cond, sigma, loss_mask=None):
        """
        Method to compute the loss for the denoiser during training adapted 
        for the Transformer model from Chi et al. 2023
        
        :param obs: obs of the environment
        :param action: action taken by the expert
        :param goal: goal of the agent
        :param noise: noise added to the action
        :param sigma: sampled noise levels
        """
        noised_input = noised_action
            
        c_skip, c_out, c_in = [append_dims(x, action.ndim) for x in self.get_scalings(sigma)]
        sigma = sigma.log() / 4
        # Compute the model output 
        pred = self.inner_model(noised_input * c_in, sigma, cond)
        # Compute the target for the denoiser using the scaling factors
        target = (action - c_skip * noised_input) / c_out
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss

    def forward(self, action, sigma, cond, ):
        """
        Method to compute the denoised action during rollout for the agent
        """
        c_skip, c_out, c_in = [append_dims(x, action.ndim) for x in self.get_scalings(sigma)]
        sigma = sigma.log() / 4
        return self.inner_model(action * c_in, sigma, cond) * c_out + action * c_skip

    def get_params(self):
        return self.inner_model.parameters()
    
    def parameters(self, recurse=True):
        for name, param in self.inner_model.named_parameters(recurse=recurse):
            yield param
    
    
