"""CLIP tokenizer for text encoding and decoding.

This module provides tokenization functionality for the CLIP model, including:
- Byte-pair encoding (BPE) tokenization
- Special token handling
- Text cleaning and normalization
- HuggingFace tokenizer wrapper

The tokenizer converts text into token sequences that can be processed by the CLIP model.
It handles unicode characters, special tokens, and provides both encoding and decoding.

Classes
-------
SimpleTokenizer
    BPE tokenizer with special token support and text cleaning.
HFTokenizer
    Wrapper for HuggingFace tokenizers with CLIP-compatible interface.

Functions
---------
default_bpe()
    Get default path to BPE vocabulary file.
bytes_to_unicode()
    Create mapping between UTF-8 bytes and unicode strings.
get_pairs(word)
    Get adjacent symbol pairs in a word.
basic_clean(text) 
    Basic text cleaning with ftfy.
whitespace_clean(text)
    Normalize whitespace in text.
decode(output_ids)
    Decode token IDs back to text.
tokenize(texts, context_length)
    Convert text to token IDs for CLIP model input.

Notes
-----
The core tokenization is based on byte-pair encoding (BPE) which:
- Handles unicode efficiently
- Provides good coverage of natural language
- Allows reversible tokenization

The tokenizer adds special tokens like <start_of_text> and <end_of_text>.
Text is cleaned and normalized before tokenization.
Maximum sequence length is typically 77 tokens for CLIP models.

Adapted from OpenAI's CLIP implementation:
https://github.com/openai/CLIP
Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import gzip
import html
import os
from functools import lru_cache
from typing import Union, List

import ftfy
import regex as re
import torch

# https://stackoverflow.com/q/62691279
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@lru_cache()
def default_bpe():
    """Get the default path to the BPE vocabulary file.
    
    Returns
    -------
    str
        Path to the gzipped BPE vocabulary file.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """Create a mapping between UTF-8 bytes and unicode strings.
    
    The BPE codes operate on unicode strings, requiring a large vocabulary
    to avoid unknown tokens. This mapping allows efficient handling of 
    UTF-8 bytes while avoiding whitespace/control characters.
    
    Returns
    -------
    dict
        Mapping from UTF-8 byte values to unicode characters.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Get all adjacent symbol pairs in a word.
    
    Parameters
    ----------
    word : tuple
        Word as a tuple of symbols (variable-length strings).
        
    Returns
    -------
    set
        Set of adjacent symbol pairs in the word.
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    """Apply basic text cleaning.
    
    Parameters
    ----------
    text : str
        Input text to clean.
        
    Returns
    -------
    str
        Cleaned text with fixed unicode and HTML entities.
    """
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    """Normalize whitespace in text.
    
    Parameters
    ----------
    text : str
        Input text to clean.
        
    Returns
    -------
    str
        Text with normalized whitespace.
    """
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    """BPE tokenizer with special token support.
    
    Implements byte-pair encoding tokenization with:
    - Special token handling
    - Unicode byte encoding
    - Text cleaning
    - Token caching
    
    Parameters
    ----------
    bpe_path : str, optional
        Path to BPE vocabulary file, by default from default_bpe()
    special_tokens : list of str, optional
        Additional special tokens beyond <start_of_text> and <end_of_text>
        
    Attributes
    ----------
    byte_encoder : dict
        Maps bytes to unicode characters.
    byte_decoder : dict
        Maps unicode characters to bytes.
    encoder : dict
        Maps tokens to token IDs.
    decoder : dict
        Maps token IDs to tokens.
    bpe_ranks : dict
        Maps BPE merges to ranks.
    cache : dict
        Cache of tokenized strings.
    vocab_size : int
        Size of the vocabulary.
    all_special_ids : list
        Token IDs of all special tokens.
    """
    def __init__(self, bpe_path: str = default_bpe(), special_tokens=None):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        if not special_tokens:
            special_tokens = ['<start_of_text>', '<end_of_text>']
        else:
            special_tokens = ['<start_of_text>', '<end_of_text>'] + special_tokens
        vocab.extend(special_tokens)
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {t:t for t in special_tokens}
        special = "|".join(special_tokens)
        self.pat = re.compile(special + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

        self.vocab_size = len(self.encoder)
        self.all_special_ids = [self.encoder[t] for t in special_tokens]

    def bpe(self, token):
        """Apply byte-pair encoding to a token.
        
        Parameters
        ----------
        token : str
            Token to encode.
            
        Returns
        -------
        str
            BPE-encoded token.
        """
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        """Encode text into token IDs.
        
        Parameters
        ----------
        text : str
            Text to encode.
            
        Returns
        -------
        list
            List of token IDs.
        """
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        """Decode token IDs back to text.
        
        Parameters
        ----------
        tokens : list
            List of token IDs.
            
        Returns
        -------
        str
            Decoded text.
        """
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text


_tokenizer = SimpleTokenizer()

def decode(output_ids: torch.Tensor):
    """Decode token IDs to text.
    
    Parameters
    ----------
    output_ids : torch.Tensor
        Tensor of token IDs.
        
    Returns
    -------
    str
        Decoded text.
    """
    output_ids = output_ids.cpu().numpy()
    return _tokenizer.decode(output_ids)

def tokenize(texts: Union[str, List[str]], context_length: int = 77) -> torch.LongTensor:
    """Convert text(s) to token IDs.
    
    Parameters
    ----------
    texts : Union[str, List[str]]
        Text or list of texts to tokenize.
    context_length : int, optional
        Maximum sequence length, by default 77.
        
    Returns
    -------
    torch.LongTensor
        Tensor of token IDs, shape [batch_size, context_length].
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<start_of_text>"]
    eot_token = _tokenizer.encoder["<end_of_text>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            tokens = tokens[:context_length]  # Truncate
            tokens[-1] = eot_token
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


class HFTokenizer:
    """HuggingFace tokenizer wrapper for CLIP compatibility.
    
    Wraps HuggingFace tokenizers to provide CLIP-compatible interface
    with same text cleaning and output format.
    
    Parameters
    ----------
    tokenizer_name : str
        Name of HuggingFace tokenizer to load.
        
    Attributes
    ----------
    tokenizer : transformers.PreTrainedTokenizer
        Underlying HuggingFace tokenizer.
    """

    def __init__(self, tokenizer_name: str):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def save_pretrained(self, dest):
        """Save tokenizer files to directory.
        
        Parameters
        ----------
        dest : str
            Destination directory.
        """
        self.tokenizer.save_pretrained(dest)

    def __call__(self, texts: Union[str, List[str]], context_length: int = 77) -> torch.Tensor:
        """Tokenize texts using HuggingFace tokenizer.
        
        Parameters
        ----------
        texts : Union[str, List[str]]
            Text or list of texts to tokenize.
        context_length : int, optional
            Maximum sequence length, by default 77.
            
        Returns
        -------
        torch.Tensor
            Tensor of token IDs, shape [batch_size, context_length].
        """
        # same cleaning as for default tokenizer, except lowercasing
        # adding lower (for case-sensitive tokenizers) will make it more robust but less sensitive to nuance
        if isinstance(texts, str):
            texts = [texts]
        texts = [whitespace_clean(basic_clean(text)) for text in texts]
        input_ids = self.tokenizer(
            texts,
            return_tensors='pt',
            max_length=context_length,
            padding='max_length',
            truncation=True,
        ).input_ids
        return input_ids
