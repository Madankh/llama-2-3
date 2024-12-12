import argparse
import os
import math
import glob 
import inspect
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import time

from typing import (AbstractSet,Collection,Dict,Iterable,List,Literal,Optional,Sequence,Tuple,Union,cast,)
import numpy
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from tiktoken.load import load_tiktoken_bpe


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

