import logging
from typing import Union, Optional, Tuple
import math 
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
import einops


logger = logging.getLogger(__name__)

# stuff from Andrew Karpathy's code adapted for Diffusion models
class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(
        self, 
        n_embd: int,
        n_heads: int,
        attn_pdrop: float,
        resid_pdrop: float,
        block_size: int,
        ):
        super().__init__()
        assert n_embd % n_heads == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )
        self.n_head = n_heads

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
    

class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(
        self, 
        n_embd: int,
        n_heads: int,
        attn_pdrop: float,
        resid_pdrop: float,
        block_size: int,
        
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(
            n_embd,
            n_heads,
            attn_pdrop,
            resid_pdrop,
            block_size,
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    

class DiffusionGPT(nn.Module):
    """the full GPT model, with a context size of block_size adapted for score-based diffusion policies"""

    def __init__(
        self,
        cond_dim: int,
        device: str,
        input_dim: int,
        output_dim: int,
        embed_dim: int,
        p_drop_emb: float,
        p_drop_attn: float,
        resid_pdrop: float,
        n_layer: int,
        n_head: int,
        n_obs_steps: int,
        horizon: int,
        linear_output = True,
    ):
        super().__init__()
        self.device = device
        self.goal_conditioned = False
        self.horizon = horizon
        goal_seq_len = 0
        # input embedding stem
        # first we need to define the maximum block size
        # it consists of the goal sequence length plus 1 for the sigma embedding and 2 the obs seq len
        block_size = goal_seq_len + 2 * n_obs_steps + 1
        # the seq_size is a little different since we have state action pairs for every timestep
        seq_size = goal_seq_len + n_obs_steps + 1
        self.tok_emb = nn.Linear(cond_dim, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))
        self.drop = nn.Dropout(p_drop_emb)
        
        # needed for calssifier free guidance learning (not used in this implementation)
        self.cond_mask_prob = 0
        
        self.action_dim = input_dim
        self.obs_dim = cond_dim
        self.embed_dim = embed_dim
        # transformer
        self.blocks = nn.Sequential(
            *[Block(
                embed_dim,
                n_head,
                p_drop_attn,
                resid_pdrop,
                block_size
            ) for _ in range(n_layer)]
        )
        # decoder head
        self.ln_f = nn.LayerNorm(embed_dim)
        
        self.block_size = block_size
        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = n_obs_steps
        # we need another embedding for the sigma
        self.sigma_emb = nn.Linear(1, embed_dim) 
        # get an action embedding
        self.action_emb = nn.Linear(input_dim, embed_dim)
        # action pred module 
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, output_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100), 
                nn.GELU(),  
                nn.Linear(100, output_dim)
            )
        # self.action_pred = nn.Linear(embed_dim, action_dim)
        
        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, DiffusionGPT):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)

    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        
        return optim_groups
    
    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def forward(
        self, 
        actions, 
        sigmas,
        cond: torch.Tensor,
        keep_last_actions: Optional[bool]=False,
        **kwargs
    ):  
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B,T',cond_dim)
        output: (B,T,input_dim)
        """
        b, t, dim = cond.size()
        sigmas = sigmas.to(self.device)
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        if len(sigmas.shape) == 0:
            sigmas = sigmas.reshape(1)
        sigmas = sigmas.expand(actions.shape[0])
        sigmas = einops.rearrange(sigmas, 'b -> b 1')
        emb_t = self.sigma_emb(sigmas.to(torch.float32))
        if len(cond.shape) == 3:
            emb_t = einops.rearrange(emb_t, 'b d -> b 1 d')
        
        # get the beginning of the state action pairs
        second_half_idx = 1
        # embed them into linear representations for the transformer
        state_embed = self.tok_emb(cond)
        action_embed = self.action_emb(actions)
        
        # if not uncond:
        if self.goal_conditioned:
            position_embeddings = self.pos_emb[
            :, :(t + self.goal_seq_len), :
            ]  # each position maps to a (learnable) vector
        else: # without goal conditioning we only have the obs sequence 
            position_embeddings = self.pos_emb[
            :, :t, :
            ]
        # note, that the goal states are at the beginning of the sequence since they are available 
        # for all states s_1, .., s_t otherwise the masking would not make sense
        state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:, :])
        # the action get the same position embedding as the related states 
        action_x = self.drop(action_embed + position_embeddings[:, self.goal_seq_len:, :])
        
        # now for the annoying part
        # we need to stack the input in the following order:
        # [sigma_emb, s_g1, .., sg_n, s_1, a_1, s_2, a_2, ..., s_n, a_n]
        # first stack actions and states in the way: [s_1, a_1, s_2, a_2, ..,]
        sa_seq = torch.stack([state_x, action_x], dim=1
                            ).permute(0, 2, 1, 3).reshape(b, 2*t, self.embed_dim)
        
        # next we stack everything together 
        input_seq = torch.cat([emb_t, sa_seq], dim=1)

        # Note we need to also adept the action masks 
        x = self.blocks(input_seq)
        x = self.ln_f(x)
        
        # now we want the last half of the output
        x = x[:, second_half_idx:, :]
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        # we need to check this for inference and adapt the max seq len accord
        if x.size()[1] < 2*self.obs_seq_len:
            x_len = int(x.size()[1]/2)
            x = x.reshape(b, x_len, 2, self.embed_dim).permute(0, 2, 1, 3)
        else:
            x = x.reshape(b, self.obs_seq_len, 2, self.embed_dim).permute(0, 2, 1, 3)
        # get the outputs related to the actions
        action_outputs = x[:, 1]
        pred_actions = self.action_pred(action_outputs)
        if keep_last_actions:
            pred_actions = torch.cat([actions[:, :-1, :], pred_actions[:, -1, :].reshape(1, 1, -1)], dim=1)

        return pred_actions
    
    def mask_cond(self, cond, force_mask=False):
        """
        Used for Classifier-free guidance to mask the goal state during training
        """
        bs, t, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones((bs, t, d), device=cond.device) * self.cond_mask_prob) # .view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            # mask = torch.bernoulli(torch.ones((bs, t, 1), device=cond.device) * self.cond_mask_prob)
            # mask = einops.repeat(mask, 'b t 1 -> b t (1 d)', d=d)
            return cond * (1. - mask)
        else:
            return cond

    def get_params(self):
        return self.parameters()

