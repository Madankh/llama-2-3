import math
import struct
import inspect
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

@dataclass
class ModelArgs:
    # Default hyperparameters for the llama 7B model
    dim:int = 4096
    n_layers : int = 32
    n_heads : int = 32
    n_kv_heads : Optional[int] = None
    vocab_size : int = 32000
    hidden_dim: Optional[int] = None
    multiple_of : int = 256
    norm_eps:float = 1e-5
    max_seq_len = 2048
    dropout:float = 0.0

class RMSNorm(torch.nn.Module):
    def __init__(self, dim:int, eps:float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        return x * torch.rsqrt(x.power(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
def precompute_freqs_cis(dim:int, end:int, theta:float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim//2)].float()/dim))
    t = torch.arange(end, device=freqs.device)
    freqs=  torch.outer(t,  freqs).float()
    freqs_cos = torch.cos(freqs) # real part
    freqs_sin = torch.sin(freqs) # imaginary part
    return  freqs_cos , freqs_sin

def reshape_for_boardcast(freqs_cis:torch.Tensor, x:torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)
    
def apply_rotary_emb(
    xq:torch.Tensor, 
    xk:torch.Tensor, 
    freqs_cos:torch.Tensor,
    freq_sin:torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    # Reshape xq and sk to meatch the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # Reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_boardcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_boardcast(freq_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freq_sin
    xq_out_i = xq_r * freqs_sin - xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freq_sin
    xk_out_i = xk_r * freqs_sin - xk_i * freqs_cos

    # flatten last two dimensions 
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out, xk_out

def repeat_kv(x:torch.Tensor, n_rep:int)->torch.Tensor:
    """
    torch.repeat_interleave(x, dim=2, repeats=n_rep)
    """
    bs,slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:,:,:, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self,args:ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = args.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim)
        self.wk = nn.Linear(args.dim , args.n_kv_heads * self.head_dim)
        self.wv = nn.Linear()