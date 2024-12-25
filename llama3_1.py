import torch
import torch.nn as nn
import math
from typing import Tuple, List, Optional
class ModelArgs:
    block_size : int = 8192
    vocab_size : int = 128256
    n_layer :int = 32
    n_head : int = 32
    n_embd : int = 4096
    n_kv_head : int = 8
    ffn_dim_multiplier : float = 1.3
    multiple_of : int = 1024
    norm_eps:float = 1e-5


# -------------------------------------------------------
# PyTorch nn.Module definitions for the lllama 3.x model
def repeat_kv(x:torch.Tensor, n_rep:int)->torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return 
    return (
        x[:,:,:,None,:]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
)

def reshape_for_broadcast(freqs_cis:torch.Tensor, x:torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_scaling(freqs:torch.Tensor):
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192 # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq/scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor)/(
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freq.dtype, device=freq.device)



class RMSNorm(nn.Module):
    def __init__(self, dim:int , eps:float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def norm(self, x):
        return x * torch.rsqrt(x.power(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        output = self.norm(x.float()).type_as(x)
        return output * self.weight
    
def percompute_freqs_cis(dim:int, end:int, theta:float, use_scaled:bool=False):
    freqs = 1.0/(theta ** torch.arange(0, dim, 2)[:(dim//2)].flaot())
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq:torch.Tensor,
                     xk:torch.Tensor,
                     freqs_cis:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_, *freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_, *freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# LLaMa building blocks
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep = self.n_head // self.n_kv_head
        self.hd = config.n_embd // config.n_head
        self.use_kv = config.use_kv
        self.flash = config.flash

        self.c_attn = nn.Linear(config.n_embd, (config.n_head + 2 * config.n_kv_head) * self.hd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # static KV cache - we could alternatively allocate it outside of the model just pass it in when needed
        if self.use_kv:
            self.cache_k = torch.zeros((config.max_gen_batch_size, config.block_size, self.n_kv_head, self.hd))
            self.cache_v = torch.zeros((config.max_gen_batch_size, config.block_size, self.n_kv_head, self.hd))
        
    def forward(self, x, freqs_cis=None, start_pos=None, mask=None):
        B, T, C = x.size()