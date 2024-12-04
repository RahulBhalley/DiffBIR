from typing import List
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .open_clip import CLIP, tokenize


class FrozenOpenCLIPEmbedder(nn.Module):
    """A frozen OpenCLIP text encoder that generates embeddings from text.

    This module uses the OpenCLIP transformer encoder to generate text embeddings.
    The weights are frozen and not updated during training.

    Args:
        embed_dim (int): Dimension of the text embeddings
        vision_cfg (dict): Vision model configuration 
        text_cfg (dict): Text model configuration
        layer (str, optional): Which transformer layer to use for embeddings. 
            Options are "last" or "penultimate". Defaults to "last".

    Attributes:
        LAYERS (List[str]): Valid options for the layer parameter
        model (CLIP): The underlying CLIP model (vision component removed)
        layer (str): Which transformer layer is being used
        layer_idx (int): Index of the transformer layer being used
    """

    LAYERS = [
        #"pooled",
        "last", 
        "penultimate"
    ]

    def __init__(self, embed_dim, vision_cfg, text_cfg, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        # model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        model = CLIP(embed_dim, dict(vision_cfg), dict(text_cfg))
        del model.visual
        self.model = model
        
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def forward(self, tokens):
        """Forward pass through the model.

        Args:
            tokens (torch.Tensor): Tokenized text input

        Returns:
            torch.Tensor: Text embeddings
        """
        z = self.encode_with_transformer(tokens)
        return z

    def encode_with_transformer(self, text):
        """Encode text using the transformer.

        Args:
            text (torch.Tensor): Tokenized text input

        Returns:
            torch.Tensor: Text embeddings of shape [batch_size, n_ctx, d_model]
        """
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
        """Forward pass through transformer blocks.

        Args:
            x (torch.Tensor): Input tensor
            attn_mask (torch.Tensor, optional): Attention mask. Defaults to None.

        Returns:
            torch.Tensor: Transformed input
        """
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text: List[str]) -> torch.Tensor:
        """Encode a batch of text strings into embeddings.

        Args:
            text (List[str]): Batch of text strings to encode

        Returns:
            torch.Tensor: Text embeddings
        """
        # convert a batch of text to tensor
        tokens = tokenize(text)
        # move tensor to model device
        tokens = tokens.to(next(self.model.parameters()).device)
        return self(tokens)
