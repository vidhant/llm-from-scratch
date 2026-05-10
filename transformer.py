from torch import nn
from attention import MultiHeadAttention
from layer_norm import LayerNorm
from feed_forward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Initialize layers relevant to the MHA module
        self.layerNorm1 = LayerNorm(cfg["emb_dim"])
        self.mha = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.dropout = nn.Dropout(cfg["drop_rate"])

        # Initialize layers relevant to the FF module
        self.layerNorm2 = LayerNorm(cfg["emb_dim"])
        self.ff = FeedForward(cfg)

    def forward(self, x):
        # x will have the shape, [batch, num_tokens, x]

        original_x = x  # shortcut connection for attention block
        x = self.layerNorm1(x)
        x = self.mha(x)
        x = self.dropout(x)
        x = x + original_x

        mhaed_x = x  # shortcut connection for feed forward block
        x = self.layerNorm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + mhaed_x

        return x
