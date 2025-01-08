import torch
import torch.nn as nn
import math
from typing import Tuple, List, Optional
import torch.nn.functional as F
import os
from pathlib import Path
import inspect
import glob
import numpy as np
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group

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
        B, T, C = x.size() # Batch size, seq, embedding dimensionalaity
        qkv = self.c_atten(x)
        q,k,v = qkv.split([self.n_head * self.hd, self.n_kv_head * self.hd, self.n_kv_head * self.hd], dim=-1)
        q ,k ,v = map(lambda x:x.view(B, T, -1, self.hd),(q,k,v)) # (B, T, NH, HD)

        q,k = apply_rotary_emb(q, k, freqs_cis)

        if self.use_kv and not self.traning and start_pos >= 0:
            self.cache_k[:B, start_pos:start_pos + T] = k
            self.cache_v[:B, start_pos:start_pos + T] = v
            k = self.cache_k[:B, : start_pos + T]
            v = self.cache_v[:B, : start_pos + T]
        
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        q, k , v = map(lambda x:x.transpose(1,2), (q,k,v)) # (B, NH, T, HD)

        if self.flash:
            y = F.scale_dot_product_attention(q, k, v, mask == 0 if T>1 else None)
        else:
            score = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.hd))
            if mask is not None:
                score = score.masked_fill(mask ==0, float('-inf'))
            att = F.softmax(score.float(), dim=-1).type_as(score)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B,T,C)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.n_embd
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
    def forward(self, x):
        x1 = self.c_fc(x)
        x2 = self.c_fc2(x)
        x2 = F.silu(x2)
        x = x1 * x2
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config.n_embd, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_enbd, config.norm_eps)
        self.mlp = MLP(config)
    def forword(self , x, freqs_cis=None, start_pos=None, mask=None):
        x = x + self.attn(self.ln1(x), freqs_cis, start_pos, mask)
        x = x + self.mlp(self.ln_2(x))
        return x
    
# @dataclass
class LlamaConfig:
    version: str = "3.1"
    block_size: int = 8192
    vocab_size: int = 128256
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: int = 8
    n_embd: int = 4096
    ffn_dim_multiplier: float = 1.3
    multiple_of: int = 1024
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    use_scaled_rope: bool = True
    max_gen_batch_size: int = 4
    use_kv: bool = True
    flash: bool = False  # use flashattention?

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        assert self.n_kv_head <= self.n_head
        assert self.n_head % self.n_kv_head == 0
        assert self.n_embd % self.n_head == 0



class LLaMa(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd, config.norm_eps),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)

        self.freqs_cis = percompute_freqs_cis(
            config.n_embd // config.n_head,
            config.block_size * 2,
            config.rope_theta, 
            config.use_scaled_rope
        )

    def forward(self, idx, targets=None, return_logits=True, start_pos=0):
        _, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the LLAMA model itself
        x = self.transformer.wte(idx) # token embeddings of shape (B, t, n_embd)
        freqs_cis = self.freqs_cis[start_pos:start_pos+t]
        mask = torch.triu(torch.ones((t, t), device=next(self.parameters()).device, dtype=torch.bool), diagonal=1)

        for i , block in enumerate(self.transformer.h):
            x = block(x, freqs_cis, start_pos, mask)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x).float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:,[-1],:]).float()
            loss = None
        if not return_logits:
            logits = None
        return logits, loss
    
    @staticmethod
    def adapt_llama_state_dict_keys(checkpoint, config: LlamaConfig):
        # Modify key names from Meta's LLaMA to our LLaMA
        # our key names are derived from GPT-2's key names
        checkpoint['transformer.wte.weight'] = checkpoint.pop('tok_embeddings.weight')

        for i in range(config.n_layer):
            for name in ['attention_norm', 'ffn_norm']:
                old_key = f'layers.{i}.{name}.weight'  # e.g. layers.x.attention_norm.weight -> transformer.h.x.ln_1.weight
                new_key = f'transformer.h.{i}.ln_{1 if name == "attention_norm" else 2}.weight'
                checkpoint[new_key] = checkpoint.pop(old_key)

        for i in range(config.n_layer):
            for name in ['attention.wq', 'attention.wk', 'attention.wv']:
                old_key = f'layers.{i}.{name}.weight'
                new_key = f'transformer.h.{i}.attn.c_attn.weight'
                if name == 'attention.wq':
                    checkpoint[new_key] = checkpoint.pop(old_key)
                else:  # merge 3 weights into transformer.h.x.attn.c_attn.weight
                    checkpoint[new_key] = torch.cat((checkpoint[new_key], checkpoint.pop(old_key)), dim=0)
            old_key = f'layers.{i}.attention.wo.weight'
            new_key = f'transformer.h.{i}.attn.c_proj.weight'
            checkpoint[new_key] = checkpoint.pop(old_key)

        ffn_map = {'w1': 'c_fc2', 'w2': 'c_proj', 'w3': 'c_fc'}
        for i in range(config.n_layer):
            for name in ['feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']:
                old_key = f'layers.{i}.{name}.weight'
                new_key = f'transformer.h.{i}.mlp.{ffn_map[name.split(".")[-1]]}.weight'
                checkpoint[new_key] = checkpoint.pop(old_key)

        checkpoint['transformer.ln_f.weight'] = checkpoint.pop('norm.weight')
        checkpoint['lm_head.weight'] = checkpoint.pop('output.weight')

        return checkpoint

    @staticmethod
    def adapt_llama_state_dict_keys_hf(checkpoint, config: LlamaConfig):
        # Modify key names from HuggingFace's LLaMA to our LLaMA
        # our key names are derived from GPT-2's key names
        checkpoint['transformer.wte.weight'] = checkpoint.pop('model.embed_tokens.weight')

        # We need to unpermute K and V because HF script permuted the original Meta-LLaMA weights
        # see: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py
        def unpermute(w, n_heads, dim1, dim2):
            return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

        for i in range(config.n_layer):
            for name in ['input_layernorm', 'post_attention_layernorm']:
                old_key = f'model.layers.{i}.{name}.weight'  # e.g. layers.x.attention_norm.weight -> transformer.h.x.ln_1.weight
                new_key = f'transformer.h.{i}.ln_{1 if name == "input_layernorm" else 2}.weight'
                checkpoint[new_key] = checkpoint.pop(old_key)

        for i in range(config.n_layer):
            for name in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']:
                old_key = f'model.layers.{i}.{name}.weight'
                new_key = f'transformer.h.{i}.attn.c_attn.weight'
                if name == 'self_attn.q_proj':
                    checkpoint[new_key] = unpermute(checkpoint.pop(old_key), config.n_head, config.n_embd, config.n_embd)
                else:  # merge 3 weights into transformer.h.x.attn.c_attn.weight
                    tensor = checkpoint.pop(old_key)
                    if name == 'self_attn.k_proj':
                        tensor = unpermute(tensor, config.n_kv_head, config.n_kv_head * (config.n_embd // config.n_head), config.n_embd)
                    checkpoint[new_key] = torch.cat((checkpoint[new_key], tensor), dim=0)
            old_key = f'model.layers.{i}.self_attn.o_proj.weight'
            new_key = f'transformer.h.{i}.attn.c_proj.weight'
            checkpoint[new_key] = checkpoint.pop(old_key)

        ffn_map = {'gate_proj': 'c_fc2', 'down_proj': 'c_proj', 'up_proj': 'c_fc'}
        for i in range(config.n_layer):
            for name in ['gate_proj', 'down_proj', 'up_proj']:
                old_key = f'model.layers.{i}.mlp.{name}.weight'
                new_key = f'transformer.h.{i}.mlp.{ffn_map[name]}.weight'
                checkpoint[new_key] = checkpoint.pop(old_key)

        checkpoint['transformer.ln_f.weight'] = checkpoint.pop('model.norm.weight')

        return checkpoint

    @classmethod
    def from_pretrained_llama3_hf(cls, model_id):
        """Loads pretrained LLaMA model weights from HuggingFace"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        assert model_id == "meta-llama/Meta-Llama-3.1-8B", "Only the 8B-base model is supported for now"
        model_args = LlamaConfig()

        model = AutoModelForCausalLM.from_pretrained(model_id)
        checkpoint = LLaMa.adapt_llama_state_dict_keys_hf(model.state_dict(), model_args)

        original_default_type = torch.get_default_dtype()  # save the default type
        torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)  # much faster loading
        model = LLaMa(model_args)
        model.load_state_dict(checkpoint, strict=False)
        torch.set_default_tensor_type(torch.tensor([], dtype=original_default_type, device="cpu").type())  # restore default type

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_id = 128004  # this is the pad token id for LLaMA 3.1 base, we need to set this explicitly as our generate func expects it
        tokenizer.stop_tokens = [tokenizer.eos_token_id]
        model.tokenizer = tokenizer
        return model

    @classmethod
    def from_pretrained_llama3_meta(cls, ckpt_dir, tokenizer_path):
        """Loads pretrained LLaMA model weights from a checkpoint directory"""
        model_args = LlamaConfig()

        ckpt_path = sorted(Path(ckpt_dir).glob("*.pth"))[0]
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        checkpoint = LLaMa.adapt_llama_state_dict_keys(checkpoint, model_args)

        original_default_type = torch.get_default_dtype()  # save the default type
        torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)  # much faster loading
        model = LLaMa(model_args)
        model.load_state_dict(checkpoint, strict=False)
        torch.set_default_tensor_type(torch.tensor([], dtype=original_default_type, device="cpu").type())  # restore default type

        tokenizer = Tokenizer(model_path=tokenizer_path)
        model.tokenizer = tokenizer
        return model
    

    def configure_optimizers(self, weight_decay, learining_rate, betas, device_type, zero_stage):
        params_dict = {pn:p for pn, p in self.parameters()}
        params_dict = {pn:p for pn, p in params_dict.items() if p.require_grad()}

        decay_params = [p for pn , p in params_dict.items() if p.dim() >=2]
        non_decay_params = [p for pn, p in params_dict.items() if p.dim() <2]

        optim_groups = [
           {"params":decay_params, "weight_decay":weight_decay},
           {"params": non_decay_params, "weight_decay":0.0}
        ]

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if zero_stage == 1:
            print0("using zeroRedundancyOptimizer")
            optimizer  = ZeroRedundancyOptimizer(**optim_groups[0], optimizer_class=torch.optim.AdamW,
                                                 lr=learining_rate, betas=betas, fused=use_fused)
            optimizer.add_param_group(optim_groups[1])
        else:
            print0("using regular AdamW")
            optimizer = torch.optim.AdamW(optim_groups, lr=learining_rate, betas=betas, fused=use_fused)
        return optimizer
    @torch.inference_mode()
    def generate(
        self, 
        promp_tokens:List[List[int]],
        max_gen_len : int,
        temperation : float = 0.6,
        top_p:float = 0.9,
        echo:bool = False
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.

        """
        bsz = len(promp_tokens)
        assert bsz <= self.config.max_gen_batch_size, f"Batch size {bsz} exceeds the maximum generation batch size {self.config.max_gen_batch_size}"
        device = next(self.parameters()).device

        min_prompt_len = min(len(p) for p in promp_tokens)
        max_prompt_len = max(len(p) for p in promp_tokens)
        assert max_prompt_len <= self.config.block_size, f"Prompt length {max_prompt_len} exceeds the maximum block size {self.config.block_size}"
        total_len = min(self.config.block_size, max_prompt_len + max_gen_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
        for idx, t in enumerate(promp_tokens):
            tokens[idx, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=device)
        input_text_mask = tokens != pad_id

        if min_prompt_len == total_len:
            logits , _ = self.forward(tokens, start_pos=prev_pos)
        
        stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens), dtype=torch.long, device=device)

        for cur_pos in range(min_prompt_len, total_len):
            logits, _ = self.forward(tokens[:,prev_pos:cur_pos], start_pos=prev_pos)
            if temperation > 0:
                probs = torch.softmax(logits[:, -1] / temperation, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)
            
            next_token = next_token.reshape(-1)

            # only replace token if prompt has already been generated
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            eos_reached |= ~input_text_mask[:, cur_pos] & torch.isin(next_token, stop_tokens)
            prev_pos = cur_pos
            if all(eos_reached):
                break

        out_tokens = []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(promp_tokens[i])
            toks = toks[start : len(promp_tokens[i]) + max_gen_len]
            # cut to after eos tok if any
            for stop_token in self.tokenizer.stop_tokens:
                try:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx]
                except ValueError:
                    pass
            out_tokens.append(toks)
        return out_tokens

def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution

    """
    probs_sort,  probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def print0(*args, **kwargs):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


# -----------------------------------------------------------------------------
# Llama 3.1 Tokenizer

# The tiktoken tokenizer can handle <=400k chars without
# pyo3_runtime.PanicException.
TIKTOKEN_MAX_ENCODE_CHARS = 400_000

# https://github.com/openai/tiktoken/issues/195
# Here we iterate over subsequences and split if we exceed the limit
# of max consecutive non-whitespace or whitespace characters.
MAX_NO_WHITESPACES_CHARS = 25_000


class Tokenizer:
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_path: str):
        """
        Initializes the Tokenizer with a Tiktoken model.

        Args:
            model_path (str): The path to the Tiktoken model file.
        """
        assert os.path.isfile(model_path), model_path

        mergeable_ranks = load_tiktoken_bpe(model_path)
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|finetune_right_pad_id|>",
            "<|step_id|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eom_id|>",  # end of message
            "<|eot_id|>",  # end of turn
            "<|python_tag|>",
        ]
        reserved_tokens = [
            f"<|reserved_special_token_{2 + i}|>"
            for i in range(self.num_reserved_special_tokens - len(special_tokens))
        ]
        special_tokens = special_tokens + reserved_tokens

        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        self.n_words: int = num_base_tokens + len(special_tokens)
        # BOS / EOS token IDs
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]
        self.eot_id: int = self.special_tokens["<|eot_id|>"]
        self.eom_id: int = self.special_tokens["<|eom_id|>"]
        self.python_tag_id = self.special_tokens["<|python_tag|>"]
        self.pad_id: int = self.special_tokens["<|finetune_right_pad_id|>"]
        # hardcoded stop tokens for the base model
        self.stop_tokens = [
            self.special_tokens["<|begin_of_text|>"],
            self.special_tokens["<|end_of_text|>"],
        ]

    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Optional[Union[Literal["all"], AbstractSet[str]]] = None,
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
            allowed_tokens ("all"|set[str]): allowed special tokens in string
            disallowed_tokens ("all"|set[str]): special tokens that raise an error when in string

        Returns:
            list[int]: A list of token IDs.

        By default, setting disallowed_special=() encodes a string by ignoring
        special tokens. Specifically:
        - Setting `disallowed_special` to () will cause all text corresponding
          to special tokens to be encoded as natural text (insteading of raising
          an error).
        - Setting `allowed_special` to "all" will treat all text corresponding
          to special tokens to be encoded as special tokens.
        """
        if allowed_special is None:
            allowed_special = set()
        assert type(s) is str

        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: List[int] = []
        for substr in substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def decode(self, t: Sequence[int]) -> str:
        # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
        return self.model.decode(cast(List[int], t))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]

# -----------------------------------------------------------------------------
# mY Own simple Distributed Data Loader

def _peek_data_shard(filename):
    # Opens binary file to read header
    with open(filename, "rb") as f:
        # Reads first 256*4 bytes as int32 array
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    
    # Checks magic number (file format identifier)
    if header[0] != 20240801:
        print("Error: magic number mismatch in the data .bin files")
        exit(1)
    
    # Verifies version number is 7
    assert header[1] == 7, "unsupported version"
    
    # Gets token count from header
    ntok = header[2]
    return ntok

def _load_data_shard(filename):
    # Opens binary file
    with open(filename, "rb") as f:
        # Reads header same as peek
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240801, "magic number mismatch"
        assert header[1] == 7, "unsupported version"
        ntok = header[2]
        
        # Reads all tokens as uint32 array
        tokens = np.frombuffer(f.read(), dtype=np.uint32)
    
    # Verifies token count matches header
    assert len(tokens) == ntok, "Token count mismatch"
    return tokens

class DistributedShardedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        # Stores process info and dimensions
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B  # batch size
        self.T = T  # sequence length
        
        # Gets all matching data files
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, "No matching files found"
        
        # Counts total tokens across all shards
        ntok_total = 0
        for fname in self.files:
            shards_ntok = _peek_data_shard(fname)
            # Verifies shard has enough tokens for distribution
            assert shards_ntok >= num_processes * B * T + 1
            ntok_total += shards_ntok
        self.ntok_total = ntok_total
        
        # Initializes shard tracking
        self.current_shard = None
        self.reset()
    
    def reset(self):
        # Resets to first shard if needed
        if self.current_shard != 0:
            self.current_shard = 0
            self.tokens = _load_data_shard(self.files[self.current_shard])
        # Sets position based on process rank
        self.current_position = self.process_rank * self.B * self.T

    def advance(self):
        # Moves to next shard cyclically
        self.current_shard = (self.current_shard + 1) % len(self.files)
        # Resets position for new shard
        self.current_position = self.process_rank * self.B * self.T
        # Loads new shard data
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        # Gets sequence of tokens for batch
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        # Converts to PyTorch tensor
        buf = torch.tensor(buf, dtype=torch.long)
        # Creates input and target tensors
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        
        # Advances position by batch size times processes
        self.current_position += B * T * self.num_processes
        
        # Moves to next shard if current exhausted
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
            
        return x, y
    
# --------------------------------------------------------------------
# Python -> C bridge utilities for saving params/grads/activations to .bin files
def write_fp32(tensor, file):
    t = tensor.detach().cpu().numpy().astype(np.float32)
    b = t.tobytes()
    file.write(b)

def write_bf16(tensor, file):
    t = tensor.detach().cpu().numpy().astype(np.bfloat16)
    t = t.view(torch.int16)
    b = t.numpy().tobytes()
    file.write(b)

def write_tensors(model_tensors, L, file, dtype):
    # writes LLaMA 3 model's weights to a binary file
    assert dtype in {"float32", "bfloat16"}
    write_fun = write_fp32 if dtype == "float32" else write_bf16
    write_fun(model_tensors["transformer.wte.weight"], file) # (V, C)
    for i in range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.weight"], file)
    for i in range(L): # (L, 3C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.weight"], file)
    for i in range(L): # (L, C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.weight"], file)
    for i in range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.weight"], file)
    for i in range(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.weight"], file)
    for i in range(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc2.weight"], file)
    for i in range(L): # (L, C, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.weight"], file)
    write_fun(model_tensors["transformer.ln_f.weight"], file) # (C, )
    write_fun(model_tensors["lm_head.weight"], file) # (V, C)


def write_model(model, filename, dtype):
    # everything we need to instantiate the model
    assert dtype in {"float32", "bfloat16"}
    version = {
        "float32":3, #  3: all tensors are fp32
        "bfloat16":5, # 5: all tensors are bf16
    }[dtype]

    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240801
    header[1] = version
    header[2] = model.config.block_size
    header[3] = model.config.vocab_size
    header[4] = model.config.n_layer
    header[5] = model.config.n_head
    header[6] = model.config.n_kv_head
    header[7] = model.config.n_embd
    header[8] = model.config.ffn_dim_multiplier
    header[9] = model.config.multiple_of
    header[10] = model.config.norm_eps
    header[11] = model.config.rope_theta
    header[12] = model.config.use_scaled_rope
    header[13] = model.config.max_gen_batch_size
    header[14] = int(model.config.version.split('.')[0])
    header[15] = int(model.config.version.split('.')[1])
    # 2) the parameters follow the header
    params = {name:params.cpu() for name, params in model.parameters()}
    # now write to file
    with open(filename, "wb") as file:
        file.write(header.numpy().tobytes()) # header
        write_tensors(params, model.config.n_layer, file, dtype) # params
    print(f"wrote {filename}")
    

if __name__ == "__main__":
    print0(f"Running pytorch {torch.version.__version__}")

    # default settings will overfit a tiny batch of data
    # and save model weights and debug state to disk on the first iteration
    parser = arg.parse.ArgumentParser()
    parser.add_argument("--use_hf", type=int, default=1, help="use HuggingFace (default) or use Meta's model")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="path to llama3 model checkpoint (needed if use_hf=0)")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="path to llama3 tokenizer (needed if use_hf=0)")
    # file system input / output
    parser.add_argument("--input_bin", type=str, default="dev/data/tinyshakespeare/tiny_shakespeare_val.bin", help="input .bin to train on")
    parser.add_argument("--input_val_bin", type=str, default="", help="input .bin to eval validation loss on")
    parser.add_argument("--output_dir", type=str, default="", help="output directory to which to write logs and checkpoints")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="chose the llama model")
    # token layout for each step of the optimization
    parser.add_argument("--batch_size", type=int, default=4, help="batch size, in units of #batch dimensions")
    parser.add_argument("--sequence_length", type=int, default=64, help="sequence length")
    parser.add_argument("--total_batch_size", type=int, default=256, help="total desired batch size, in units of #tokens")
    # workload (number of steps)
    parser.add_argument("--num_iterations", type=int, default=10, help="number of iterations to run")
    parser.add_argument("--inference_only", type=int, default=0, help="only run inference")
    # optimization
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate warmup iterations")
    parser.add_argument("--warmup_iters", type=int, default=0, help="learning rate warmup iterations")
    parser.add_argument("--learning_rate_decay_frac", type=float, default=1.0, help="learning rate warmup iterations")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="maximum gradient magnitude")
    # evaluation
    parser.add_argument("--val_loss_every", type=int, default=0, help="every how mant steps to evaluate val loss?")
    parser.add_argument("--val_max_steps", type=int, default=20, help="how many batches of val to average?")
    parser.add_argument("--sample_every", type=int, default=0, help="how often to sample from the model?")
    # debugging
    parser.add_argument("--overfit_single_batch", type=int, default=1, help="overfit just one batch of data")
    # numerics
    parser.add_argument("--tensorcores", type=int, default=0, help="use tensorcores")
    # memory management
    parser.add_argument("--device", type=str, default="", help="by default we autodetect, or set it here")
    parser.add_argument("--compile", type=int, default=0, help="torch.compile the model")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|float16|bfloat16")
    parser.add_argument("--zero_stage", type=int, default=0, help="zero redundancy optimizer stage (0/1/2/3)")
    # python -> C bridge
    parser.add_argument("--write_tensors", type=int, default=0, help="write tensors to disk")
    args = parser.args()

    # args error checking and convenience variables
    B, T = args.batch_size, args.sequence_length
    assert 1 <= T <= 8192, "sequence length must be between 1 and 8192"
    assert args.dtype in {"float32", "float16", "bfloat16"}
    assert args.model in {"meta-llama/Meta-Llama-3.1-8B"}  # only 8B base model supported for no
    
    # create the logging directory if it does not exist
    logfile = None
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logfile = os.path.join(args.output_dir, "main.log")
        # create the log file "main.log" inside it and wipe it clean
        with open(logfile, "w") as f:
            pass


    # set up DDP (distributed data parallel). torchrun sets this env variable
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this ddp run
    if ddp:
        # use of DDP atm demands cuda we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for noowo i think we need cuda for ddp"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging , checkpointing etc.
        seed_offset = 0 # each process gets the exact same seed
        zero_stage = args.zero_stage
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        zero_stage = 0
        ddp_world_size = 1
        master_process = True
        seed_offset = 0

        if args.device:
            device = args.device
        else:
            # attemt to autodetect the device
            device = 'cpu'
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
    device_type = 'cuda' if 'cuda' is device else 'cpu'
    assert device_type in {"cuda"}, "GPU required to run LLAMA 3"
    print(f"using device : {device}")

    # calculate gradient accumulation from the desired total batch size and the current run configuration
    token_per_fwdbwd = B * T * ddp_world_size
    assert args.total_batch_size % token_per_fwdbwd == 0
    grad_accum_steps = args.total_batch_size // token_per_fwdbwd
    print0(f"total desired batch size: {args.total_batch_size}")
    print0(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # set up a context manager following the desired dtype and device
    ptdtype = {'float32':torch.float32, 'bfload16':torch.bfloat16, 'float16':torch.float16}[args.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if (device_type == "cuda") else nullcontext()
    #rng / reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # set the torch precision mode to use TensorFloat32 (TF32) for matmuls
    # docs https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    if args.tensorcores:
        torch.set_float32_matmul_precision('high')
    
    # init the model
    if args.use_hf:
        model = LLaMa.from_pretrained_llama3_hf(args.model)
    else:  # use Meta's checkpoint
        assert args.ckpt_dir is not None and os.path.exists(args.ckpt_dir), f"llama3 ckpt dir {args.ckpt_dir} does not exist"
        assert args.tokenizer_path is not None and os.path.exists(args.tokenizer_path), f"llama3 tokenizer path {args.tokenizer_path} does not exist"
        model = LLaMa.from_pretrained_llama3_meta(args.ckpt_dir, args.tokenizer_path)

    model.train()
    if args.compile:
        if hasattr(config, "coordidnate_descent_tuning"):
            config.coordinate_descent_tuning = True
        print0("Compiling the model..")
        model = torch.compile(model)

    # Our own version of simple DistrubutedDataloader
    # # load tokens
    train_loader = DistributedShardedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = None
    if args.input_val_bin:
        val_loader = DistributedDataParallel(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
    # PyTorch -> C bridge: save some weights and state for C to load later as reference
    # 
    # do one forward pass to genereate ground truth for our C tests
    #     


