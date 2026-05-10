import torch
from torch import nn
from transformer import TransformerBlock
from layer_norm import LayerNorm


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.sem_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # bias=False as we often tie these weights with sem_emb and to improve training stability
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        if cfg["weight_tying"]:
            self.out_head.weight = self.sem_emb.weight

    def forward(self, input):
        batch_size, seq_len = input.shape
        sem_embeds = self.sem_emb(input)
        # Create position indices [0, 1, ..., seq_len-1] and
        # look up their learned embeddings
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=input.device))

        x = sem_embeds + pos_embeds
        x = self.dropout(x)

        x = self.transformer_blocks(x)
        x = self.final_norm(x)

        logits = self.out_head(x)

        return logits
