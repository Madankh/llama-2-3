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
