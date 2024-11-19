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

class RMSNorm(nn.Module):
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

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)

        self.wk = nn.Linear(args.dim , args.n_kv_heads * self.head_dim, bias=False)

        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)

        self.wo = nn.Linear(args.n_heads*self.head_dim, args.dim, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)

        self.resid_dropout = nn.Dropout(args.dropout)

        self.dropout = args.dropout

        # use flash attention or a manual implementatioon
        if not self.flash:

            print("WARNING: using slow attention, flash attention require pytorch >= 2.0")

            mask = torch.full((1,1,args.max_seq_len, args.max_seq_len))

            mask = torch.triu(mask, diagonal=1)

            self.register_buffer("mask", mask)
        
        def forward(
                self, 
                x:torch.Tensor, 
                freqs_cos:torch.Tensor,
                freqs_sin:torch.Tensor,
            ):

            bsz, seqlen = x.shape
            # QKV
            xq,xk,xv = self.wq(x), self.wk(x), self.wv(x)

            xq = xq.view(bsz, seqlen , self.n_local_heads, self.head_dim)

            xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

            xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

            # Rope relative position embedding

            xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

            # Grouped multiquery attention: expand out keys and values
            xk = repeat_kv(xk, self.n_rep) # (bs, seqlen, n_local_heads, head_dim)

            xv = repeat_kv(xv, self.n_rep)

            # make heads into batch dimension
            xq = xq.transpose(1,2) # (bs, n_local_heads, seqlen, head_dim)

            xk = xk.transpose(1,2)

            xv = xv.transpose(1,2)

            # flash implementation
            if self.flash:
                output = torch.nn.functional.scaled_dot_product_attention(xq, xk ,xv, attn_mask=None, dropout_p=self.dropout if self.traning else 0.0, is_causal=True)
            else:
                # manual implementation
                scores = torch.matmul(xq, xk.transpose(2,3)) / math.sqrt(self.head_dim)

                assert hasattr(self, 'mask')

                scores = scores + self.mask[:,:,:seqlen, :seqlen]  # (bs, n_local_heads, seqlen, cache_len + seqlen)

                scores = F.softmax(scores.float(), dim=-1).type_as(xq)
                
                scores = self.attn_dropout(scores)

                output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

            # restore time as batch dimension and concat heads
            output = output.transpose(1,2).contiguous().view(bsz, seqlen, -1)

            # final projection into residual stream
            output = self.wo(output)

            output = self.resid_dropout(output)

            return output

        
class FeedForward(nn.Module):

    def __init__(self, dim:int, hidden_dim:int, mutiple_of:int , dropout:float):

        super().__init__()

        if hidden_dim is None:
           
           hidden_dim = 4 * dim

           hidden_dim = int(2*hidden_dim/3)

           hidden_dim = mutiple_of * ((hidden_dim + mutiple_of - 1) // mutiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)

        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class TransformerBlock(nn.Module):

    def __init__(self, layer_id:int, args:ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            mutiple_of=args.multiple_of,
            dropout=args.dropout
        )
        self.layer_id = layer_id,
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Normalization Before the feed forward block
        self.ffn_norm = RMSNorm(args.dim, eps = args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        # (B, Seq_len, Dim)
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
    
class Transformer(nn.Module):
    last_loss : Optional[torch.Tensor]

    def __init__(self, params:ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm =  RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False))

        # share the unemdedding parameters with the embedding parameters
        self.tok_embeddings.weight = self.output.weight

        # some useful precompute for the RoPE relative positional embedding
        freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim//self.params.n_heads, self.params.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2*params.n_layers))
        # Initialize attribute for the loss of the last forward call. This will be set if the forward is called with a targets tensor.
        self.last_loss = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, tokens:torch.Tensor, targets:Optional[torch.Tensor] = None):
        _bsz, seqlen =  tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        for layer in self.layers:
            h = layer(TransformerBlock(h, freqs_cos, freqs_sin))
        h = self.norm(h)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.output(h[:,[-1],:]) # note: using list [-1] to preserve the time dim
            self.last_loss = None
        return logits
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        params_dict = {pn : p for pn, p in self.parameters()}
        # filter out those that do not require grad
        params_dict = {pn : p for pn , p in params_dict.items() if p.requires_grad}
        # create optim groups. any parameters that is 2d will be weight decayed , otherwise no 
        # i.e all weight tensors in matmuls + embeddings decay , all biases and layernorms don't 
        decay_params = [p for n , p in params_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in params_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params":decay_params, "weight_decay":weight_decay},
            {"params":nodecay_params, "weight_decay" : 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors : {len(decay_params)}")
        print(f"num nodecayed parameters tensor : {len(nodecay_params)}")

        # Create AdamW optimizer and use the fused version if it is available
        fused_avaiable = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_avaiable and device_type=='cuda'
        extra_args = dict(fused = True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, **extra_args)
        return optimizer
    
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS
        """
        # First estimate the number of flops we do per iteration
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in self.parameters())
        cfg = self.params
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim//cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd  =  flops_per_token * T
        flops_per_iter  = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved  = flops_per_iter * (1.0/dt) # per second

        flops_promised = 312e12 
        mfu = flops_achieved / flops_promised
        return mfu
    
    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        Also note this is a super inefficient version of sampling with no key/value cache.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            logits = logits[:, -1, :] # crop to just the final time step
            if temperature == 0.0:
                # "sample" the single most likely index
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # pluck the logits at the final step and scale by desired temperature
                logits = logits / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx