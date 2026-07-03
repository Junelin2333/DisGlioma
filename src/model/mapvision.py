import torch
import torch.nn as nn
import os
from einops import rearrange, repeat
from typing import Optional
from torch.nn import functional as F
from transformers import AutoModel
from collections import OrderedDict
from DisGlioma.src.ResNet.encoder import VisionEncoder
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VisionConfig:
    num_cluster: int = 3
    num_prompts: int = 4
    num_classes: int = 4
    proj_dim: int = 128
    center_dim: int = 256
    embed_dim: int = 768
    llm_model: str = "emilyalsentzer/Bio_ClinicalBERT"
    cross_attn: bool = True
    vision_pretrain: bool = True
    confidence_threshold: float = 0.85


class VisionModel(nn.Module):
    def __init__(self, config:Optional[VisionConfig] = None):
        super().__init__()
        ### config LLM ###
        config = config if config else VisionConfig()
        self.config = config
        self.num_cluster = config.num_cluster
        self.num_prompt = config.num_prompts

        self.llm = AutoModel.from_pretrained(config.llm_model)
        for params in self.llm.parameters():
            params.requires_grad = False

        #### config gene prompt ###
        self.gene_prompt = nn.Parameter(torch.empty((config.num_cluster, config.num_prompts, config.embed_dim)), requires_grad=False)
        prompt_weights = torch.load(Path(__file__).resolve().parent.parent / 'ckpt' / 'prompt_st3.pt', map_location='cpu')
        self.gene_prompt.data.copy_(prompt_weights)

        self.proj = nn.Sequential(
                nn.Linear(config.embed_dim, config.center_dim),
                nn.GELU(),
                nn.RMSNorm(config.center_dim),
            )

        ### config visual model ###
        self.vision_encoder = VisionEncoder()
        self.norm1 = nn.RMSNorm(config.center_dim)
        
        ### config multimodal cross attention ###
        if config.cross_attn:
            if config.vision_pretrain:
                self._load_vision()
            self.attn1 = nn.MultiheadAttention(config.embed_dim, num_heads=2, batch_first=True)
            self.up = nn.Sequential(
                nn.Linear(128, config.embed_dim),
                nn.GELU(),
                nn.RMSNorm(config.embed_dim)
            )

            self.pred = nn.Sequential(
                nn.Linear(config.embed_dim, config.proj_dim),
                nn.SiLU(),
                nn.Linear(config.proj_dim, config.num_classes),
            )
            
        else:
            self.pred = nn.Sequential(
                nn.Linear(config.proj_dim, config.num_classes),
            )
        
    def forward(self, data):
        hazard_embed, cls_embed = self.vision_encoder.forward(data['image'])
        bsz = cls_embed.shape[0]
        cls_embed = self.norm1(cls_embed)

        gene_embed = repeat(self.gene_prompt, 'n l c -> b n l c', b=bsz)
        text_embed = self.get_cls_embedding(data['text_ids'], gene_embed)
        sim = self.similarity(cls_embed.unsqueeze(dim=1), text_embed)

        if self.training:
            c = data['cluster']
            for i in range(bsz):
                if c[i] !=-1:
                    continue
                else:
                    c[i] = torch.argmax(sim[i], dim=-1)
            mask = F.one_hot(c, num_classes=self.config.num_cluster)
            mask = mask.unsqueeze(dim=1).to(torch.float)
        else:
            mask = torch.where(sim > self.config.confidence_threshold, 1., 0)

        if self.config.cross_attn:
            gene_embed = rearrange(gene_embed, 'b n l c -> b n (l c)')
            gene_embed = torch.bmm(mask, gene_embed)
            gene_embed = rearrange(gene_embed.squeeze(dim=1), 'b (l c) -> b l c', l=4)

            h_embed = self.up(hazard_embed)
            attn, _ = self.attn1(h_embed.unsqueeze(dim=1), gene_embed, gene_embed)

            h_embed = h_embed + attn.squeeze(dim=1)
            hazard = self.pred(h_embed)
            return sim.squeeze(dim=1), hazard, h_embed
        
        else:
            hazard = self.pred(h_embed)
            return sim.squeeze(dim=1), hazard, cls_embed

    def get_cls_embedding(self, token_ids, cls_embed=None):
        ## cls_embed: [bsz, num_class, num_prompt, embed_dim]
        token_ids = rearrange(token_ids, 'b n l -> (b n) l')
        token_embed = self.llm.embeddings(token_ids)  # [(bsz * n), l , embed_dim]
        
        prefix = token_embed[:, :1, :]   # CLS
        suffix = token_embed[:, 1:, :]   # caption body + SEP/PAD

        cls_embed = rearrange(cls_embed, 'b n l c -> (b n) l c')

        prompt_embed = torch.cat([prefix, cls_embed, suffix], dim=1)

        o = self.llm.forward(inputs_embeds=prompt_embed, output_hidden_states=True)
        pool = o['pooler_output']
        proj = self.proj(pool)
        proj = rearrange(proj, '(b n) c -> b n c', n=self.num_cluster)
        
        return proj

    def similarity(self, visual_embed, text_embed):
        p_norm = F.normalize(visual_embed, p=2, dim=-1)   # [bsz, embed_dim]
        t_norm = F.normalize(text_embed, p=2, dim=-1)    # [bsz, 3, embed_dim]

        sim = torch.bmm(p_norm, t_norm.transpose(1, 2))
        return sim


    def _load_vision(self):
        ckpt_path = (Path(__file__).resolve().parents[2] / 'save_model' / 'map_vision.pt')
        if not ckpt_path:
            return
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"vision_pretrain not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)

        new_ckpt = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("model.vision_encoder.", "")
            new_ckpt[new_key] = v

        msg = self.vision_encoder.load_state_dict(new_ckpt, strict=True)
        print(
            "Loaded vision encoder weights:",
            f"missing={msg.missing_keys}",
            f"unexpected={msg.unexpected_keys}",
        )