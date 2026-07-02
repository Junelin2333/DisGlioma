import torch
import torch.nn as nn
import torch.nn.functional as F
from DisGlioma.src.scGPT import scGPTEmbedding
from DisGlioma.src.PathBINN.layers import PathwayMoE, GraphConv
from dataclasses import dataclass
from einops import repeat, rearrange
from typing import Union, Sequence, Optional
from pathlib import Path


@dataclass
class PathwayEncoderConfig:
    # For Gene Encoder
    vocab_size: int = 60697
    embed_size: int = 512
    pad_token_id: int = 0
    input_emb_style: str = "continuous"
    dropout: float = 0.1
    pretrained: str = Path(__file__).resolve().parent.parent / "scGPT" / "all33M_encoder.bin"
    # For MoE
    num_prompts: int = 1
    num_pathway: Sequence = 331
    # For Survival Model
    proj_dim: int = 128
    graph_dropout: float = 0.
    num_classes: int = 4


class PathwayEncoder(nn.Module):
    def __init__(self, config:Optional[PathwayEncoderConfig]=None) -> None:
        super(PathwayEncoder, self).__init__()
        ####---- init PathwayMoE ----####
        config = PathwayEncoderConfig() if config is None else config
    
        self.embedding = scGPTEmbedding(config) 
        if config.pretrained is not None:
            model_weights = torch.load(config.pretrained, map_location="cpu")
            self.embedding.load_state_dict(model_weights)
            for params in self.embedding.parameters():  # value_encoder
                params.requires_grad = False        

        self.value_encoder = nn.Sequential(
            nn.Linear(1, config.embed_size // 2),
            nn.GELU(),
            nn.Linear(config.embed_size // 2, config.embed_size),
            nn.RMSNorm(config.embed_size),
        )

        self.expert = PathwayMoE(config.embed_size, config.num_pathway, config.num_prompts)
        self.gcn = GraphConv(config.embed_size, config.proj_dim, config.graph_dropout)

        ####---- task head ----####
        self.proj1 = nn.Sequential(
            nn.Conv1d(config.num_pathway,1,1,1),
            nn.SiLU(),
            nn.Linear(config.embed_size, config.proj_dim),
            nn.GELU(),
        )

        self.pred = nn.Sequential(
            nn.Linear(config.proj_dim, config.num_classes),
        )

    def forward(self, input):
        input_ids, values = input['input_ids'], input['values']
        gene_embed = self.embedding.forward(input_ids)   # [bsz,seq_l,dim]
        value_embed = self.value_encoder(values.unsqueeze(dim=-1))

        router = input['router']
        adj_matrix = input['adj_matrix']
        
        o = self.expert.forward(gene_embed, value_embed, router)
        go = self.gcn(o, adj_matrix)  
        go = o + go

        proj1 = self.proj1(go)
        proj1 = rearrange(proj1, 'b c l -> b (c l)')
   
        pred = self.pred(proj1)

        return pred, go, proj1

