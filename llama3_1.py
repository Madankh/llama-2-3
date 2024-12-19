import torch
import torch.nn as nn
import math

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
    
def percompute_theta_pos_freqs(head_dim:int, seq_len:int, device:str, theta:float=10000.0):
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    m = torch.arange(seq_len, device=device)
    freqs = torch.outer(m, theta)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


    

