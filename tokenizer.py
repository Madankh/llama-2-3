import os
import struct
import argparse
from typing import List

from sentencepiece import SentencePieceProcessor

TOKENIZER_MODEL = "tokenizer.model" # the llama sentencepices tokenizer model

class Tokenizer:
    def __init__(self, tokenizer_model=None):
        model_path = tokenizer_model if tokenizer_model else TOKENIZER_MODEL
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS/EOS token IDs
        self.n_words:int = self.sp_model.vocab_size()
        self.bos_id:int = self.sp_model.bos_id()
        self.eos_id:int = self.sp_model.eos_id()
        self.pad_id:int = self.sp_model.pad_id()
         #print(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")
    def encode(self, s:str, bos:bool, eos:bool)-> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id] + t
        return t
    def decode(self, t:List[int])-> str:
        return self.sp_model.decode(t)
    
    def export(self):
        # get all the token aand their scores as floats
        tokens, score = [], []