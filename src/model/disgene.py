import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from DisGlioma.src.PathBINN import PathwayEncoder
from transformers import AutoModel
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class DisGeneConfig:
    embed_size: int = 512
    num_pathway: int = 331
    binn_ckpt: str = (Path(__file__).resolve().parents[2] / "save_model" / "path_binn.pt").resolve()

    num_cluster: int = 3
    num_prompts: int = 4
    num_classes: int = 4
    
    llm_embed: int = 768
    llm_model: str = "emilyalsentzer/Bio_ClinicalBERT"

    proj_dim: int = 128

class CustomCLIP(nn.Module):
    def __init__(self, config:Optional[DisGeneConfig]=None):
        super().__init__()
        config = config if config else DisGeneConfig()
        self.num_cluster = config.num_cluster
        self.num_prompt = config.num_prompts

        self.pathway_encoder = PathwayEncoder()
        self.load_ckpt(config.binn_ckpt)

        self.up_proj = nn.Sequential(
            nn.Linear(config.embed_size, config.llm_embed),
            nn.GELU(),
        )

        self.llm = AutoModel.from_pretrained(config.llm_model)
        for params in self.llm.parameters():
            params.requires_grad = False

        self.embedding = self.llm.embeddings
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

        self.llm_proj = nn.Sequential(
            nn.Linear(config.llm_embed, config.proj_dim),
            nn.GELU(),
        )

        self.cluster_embed = nn.Parameter(torch.empty(self.num_cluster, self.num_prompt, config.llm_embed), requires_grad=True)
        nn.init.normal_(self.cluster_embed, std=0.02)

        self.attn = CrossAttention(config.llm_embed, 2)

        self.pred = nn.Sequential(
            nn.Conv1d(self.num_prompt,1,1,1),
            nn.Linear(config.llm_embed, config.proj_dim),
            nn.GELU(),
            nn.Linear(config.proj_dim, config.num_classes),
        )
    def forward(self, data):
        _, path_feat, path_proj = self.pathway_encoder.forward(data)
        bsz, _, _ = path_feat.shape
        path_feat = self.up_proj(path_feat)  # align with llm embed

        if self.training:
            cls_embed = repeat(self.cluster_embed, 'n l c -> b (n l) c', b=bsz)
            c = data['cluster']

            cls_embed = self.attn.forward(cls_embed, path_feat)
            cls_embed = rearrange(cls_embed, 'b (n l) c -> b n l c', n=self.num_cluster)

            text_proj = self.get_cls_embedding(data['text_ids'], data['text_mask'], cls_embed)
            sim = self.similarity(path_proj.unsqueeze(dim=1), text_proj)

            mask = torch.argmax(sim, dim=-1)
            mask = F.one_hot(mask, self.num_cluster)
            mask = mask.float()

            cls_embed = rearrange(cls_embed, 'b n l c -> b n (l c)')
            hazard_embed = torch.bmm(mask, cls_embed)
            hazard_embed = rearrange(hazard_embed.squeeze(dim=1), 'b (l c) -> b l c', l=self.num_prompt)
            hazard = self.pred(hazard_embed)
        
        else:
            cls_embed = repeat(self.cluster_embed, 'n l c -> b (n l) c', b=bsz)
            cls_embed = self.attn.forward(cls_embed, path_feat)
            cls_embed = rearrange(cls_embed, 'b (n l) c -> b n l c', n=self.num_cluster)

            text_proj = self.get_cls_embedding(input_ids=data['text_ids'], attn_mask=data['text_mask'], cluster_embed=cls_embed)
            sim = self.similarity(path_proj.unsqueeze(dim=1), text_proj)
            
            mask = torch.argmax(sim, dim=-1)
            mask = F.one_hot(mask, self.num_cluster)
            mask = mask.float()

            cls_embed = rearrange(cls_embed, 'b n l c -> b n (l c)')
            hazard_embed = torch.bmm(mask, cls_embed)
            hazard_embed = rearrange(hazard_embed.squeeze(dim=1), 'b (l c) -> b l c', l=self.num_prompt)
            hazard = self.pred(hazard_embed)

        return sim.squeeze(dim=1), hazard.squeeze(dim=1)

    def get_cls_embedding(self, input_ids, attn_mask, cluster_embed=None):
        if cluster_embed is None:
            cluster_embed = self.cluster_embed  
        ## cls_embed: [bsz, num_class, num_prompt, embed_dim]
        
        input_ids = rearrange(input_ids, 'b n l -> (b n) l')
        attn_mask = rearrange(attn_mask, 'b n l -> (b n) l')
        token_embed = self.embedding(input_ids)  # [(bsz * n), l , embed_dim]
        cluster_embed = rearrange(cluster_embed, 'b n l c -> (b n) l c')
        
        prefix = token_embed[:, :1, :]   # CLS
        suffix = token_embed[:, 1:, :]   # caption body + SEP/PAD

        prompt_embed = torch.cat([prefix, cluster_embed, suffix], dim=1)

        prefix_mask = attn_mask[:, :1]
        suffix_mask = attn_mask[:, 1:]

        prompt_mask = torch.ones(
            cluster_embed.shape[0],
            cluster_embed.shape[1],
            device=input_ids.device,
            dtype=attn_mask.dtype,
        )

        attention_mask = torch.cat([prefix_mask, prompt_mask, suffix_mask], dim=1)

        o = self.llm.forward(
            inputs_embeds=prompt_embed,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        pool = o['pooler_output']

        pool = self.llm_proj(pool)
        pool = rearrange(pool, '(b n) c -> b n c', n=self.num_cluster)

        return pool

    def similarity(self, pathway_emb, text_embed):
        p_norm = F.normalize(pathway_emb, p=2, dim=-1)   # [bsz, embed_dim]
        t_norm = F.normalize(text_embed, p=2, dim=-1)    # [bsz, 3, embed_dim]

        sim = torch.bmm(p_norm, t_norm.transpose(1, 2))
        logit_scale = self.logit_scale.exp().clamp(max=100)
        
        return logit_scale * sim

    def load_ckpt(self, ckpt_path:str=None):
        # ckpt_path = Path(__file__).resolve().parent / "PathBINN" / "pretrained.pt"
        ckpt = torch.load(ckpt_path,map_location="cpu")
        self.pathway_encoder.load_state_dict(ckpt, strict=True)

        for params in self.pathway_encoder.parameters():
            params.requires_grad = False


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # 3. FFN (前馈神经网络)，进一步提炼注意力输出的特征
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        self.norm1 = nn.RMSNorm(embed_dim)
        self.norm2 = nn.RMSNorm(embed_dim)

    def forward(self, query, kv):

        attn_out, _ = self.cross_attn(query, kv, kv)
        x = self.norm1(query + attn_out)
        x = self.norm2(x + self.ffn(x))

        return x

    
