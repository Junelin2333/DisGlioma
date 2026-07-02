import torch.nn as nn
import torch
from typing import Optional

class scGPTEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gene_encoder = nn.ModuleDict({
            "embedding":
                nn.Embedding(config.vocab_size,
                             config.embed_size,
                             padding_idx=config.pad_token_id),
            "enc_norm":
                nn.LayerNorm(config.embed_size)
        })
        # Value encoder remains the same
        if config.input_emb_style == "continuous":
            self.value_encoder = nn.ModuleDict({
                "linear1": nn.Linear(1, config.embed_size),
                "linear2": nn.Linear(config.embed_size, config.embed_size),
                "activation": nn.ReLU(),
                "norm": nn.LayerNorm(config.embed_size),
                "dropout": nn.Dropout(config.dropout)
            })
        elif config.input_emb_style == "scaling":
            self.value_encoder = nn.Identity()
            raise Exception(
                "scaling input embedding style not supported because this model was trained on continuous style"
            )
        else:
            raise Exception("unsupported embedding style")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,) -> torch.Tensor:
        # Gene embeddings
        gene_emb = self.gene_encoder["embedding"](input_ids)
        gene_emb = self.gene_encoder["enc_norm"](gene_emb)

        return gene_emb
        

