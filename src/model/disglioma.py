import torch
import torch.nn as nn
import os
from einops import rearrange, repeat
from typing import Optional
from torch.nn import functional as F
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from DisGlioma.src.model.mapvision import VisionModel
from DisGlioma.src.model.layers.transformer_decoder import TransformerDecoder
from DisGlioma.src.utils.transform import (
    clinical_variable_status_token_list,
    clinical_variable_token_list,
    mapping_dict,
)

@dataclass
class DisGlioConfig:
    num_cluster: int = 3
    num_prompts: int = 4
    num_classes: int = 4
    proj_dim: int = 128
    embed_dim: int = 768
    llm_model: str = "emilyalsentzer/Bio_ClinicalBERT"

    language_feature_init: str = "auto"

    vision_embed_dim: int = 768
    vision_pretrain: int = Path(__file__).resolve().parents[2] / 'save_model' / "attn_vision.pt"
    freeze_vision_encoder: bool = True
    
    out_bins: int = 4
    clinical_embed_dim: int = 768
    adapter_bottleneck_dim: int = 96  #16

    decoder_embed_dim: int = 256
    decoder_depth: int = 2  
    decoder_num_heads: int = 4
    decoder_mlp_ratio: float = 4.0
    decoder_qkv_bias: bool = True
    decoder_drop_rate: float = 0.0
    decoder_attn_drop_rate: float = 0.0
    decoder_cross_attn_drop_rate: float = 0.1
    decoder_drop_path_rate: float = 0.0


class DisGlioma(nn.Module):
    def __init__(self, config:Optional[DisGlioConfig] = None):
        super(DisGlioma, self).__init__()
        ### config LLM ###
        config = config if config else DisGlioConfig()
        self.config = config
        self.num_cluster = config.num_cluster
        self.num_prompt = config.num_prompts

        self.language_features = self._build_language_features()

        ### config visual model ###
        self.vision_encoder = VisionModel()
        self._maybe_load_vision_pretrain()
        if self.config.freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        self.vision_adapter = nn.Sequential(
            nn.Linear(self.config.vision_embed_dim, self.config.decoder_embed_dim),
            nn.GELU(),
        )

        self.clinical_adapter = nn.Sequential(
            nn.Linear(self.config.clinical_embed_dim, self.config.decoder_embed_dim),
            nn.GELU(),
        )
        
        max_encoder_length = (
            1 + len(clinical_variable_status_token_list)
        )

        self.decoder = TransformerDecoder(
            embed_dim=self.config.decoder_embed_dim,
            depth=self.config.decoder_depth,
            num_heads=self.config.decoder_num_heads,
            mlp_ratio=self.config.decoder_mlp_ratio,
            qkv_bias=self.config.decoder_qkv_bias,
            drop_rate=self.config.decoder_drop_rate,
            attn_drop_rate=self.config.decoder_attn_drop_rate,
            cross_attn_drop_rate=self.config.decoder_cross_attn_drop_rate,
            drop_path_rate=self.config.decoder_drop_path_rate,
            num_tokens=self.config.out_bins,
            num_classes=1,
            max_encoder_length=max_encoder_length,
        )


    def forward(self, data, clinical:Optional[torch.Tensor]=None):
        vision_output = self.forward_vision(data)
        status_output = self.forward_language(
            clinical,
            clinical_variable_status_token_list,
            self.clinical_adapter,
        )

        combined_features = torch.cat(
            [vision_output["features"],
             status_output['features']], dim=1,
        )
        combined_attention_mask = torch.cat(
            [vision_output["attention_mask"], 
             status_output['attention_mask']], dim=1
        )

        hazards = self.decoder(
            encoder_out=combined_features,
            encoder_mask=combined_attention_mask,
            save_attn=False,
        )

        return vision_output['logits'], hazards, vision_output['cls_embed']
    
    def forward_vision(self, image):
        logits, h_embed, cls_embed = self.vision_encoder.forward(image)
        h_embed = self.vision_adapter(h_embed).unsqueeze(1)
        attention_mask = torch.ones(h_embed.size(0), 1, dtype=torch.long, device=h_embed.device)
        
        return {
            "features": h_embed,
            "cls_embed": cls_embed.unsqueeze(1),
            "attention_mask": attention_mask,
            "logits": logits,
        }

    def _build_language_features(self):
        init_mode = self.config.language_feature_init.lower()
        loaded_features = None

        if init_mode in {"auto", "bert"}: 
            try:
                from DisGlioma.src.model.layers.init_language_features import init_language_features

                loaded_features = init_language_features()
                print("Loaded text-initialized clinical features.")
            except Exception as exc:
                if init_mode == "bert":
                    raise RuntimeError("Failed to initialize clinical features from text encoder.") from exc
                print(f"Falling back to random clinical features: {exc}")

        feature_bank = nn.ModuleDict()
        for var_name in clinical_variable_token_list:
            params = nn.ParameterDict()
            for key in [*mapping_dict[var_name].keys(), -1]:
                init_tensor = None
                if (
                    loaded_features is not None
                    and var_name in loaded_features
                    and key in loaded_features[var_name]
                    and loaded_features[var_name][key].shape[-1] == self.config.clinical_embed_dim
                ):
                    init_tensor = loaded_features[var_name][key].detach().clone()
                if init_tensor is None:
                    init_tensor = torch.randn(1, self.config.clinical_embed_dim) * 0.02
                params[str(key)] = nn.Parameter(init_tensor, requires_grad=True)
            feature_bank[var_name] = params
        return feature_bank

    
    def forward_language(self, clinical_variables, variable_names, adapter):
        device = next(self.parameters()).device
        batch_size = len(next(iter(clinical_variables.values())))
        clinical_features = torch.zeros(
            batch_size,
            len(variable_names),
            self.config.clinical_embed_dim,
            device=device,
        )
        attention_mask = torch.zeros(batch_size, len(variable_names), dtype=torch.long, device=device)

        for var_idx, var_name in enumerate(variable_names):
            values = clinical_variables[var_name]
            if isinstance(values, torch.Tensor):
                value_list = values.detach().cpu().tolist()
            else:
                value_list = list(values)

            feature_list = []
            mask_list = []
            for value in value_list:
                key = int(value)
                feature_list.append(self.language_features[var_name][str(key)])
                mask_list.append(0 if key == -1 else 1)

            clinical_features[:, var_idx, :] = torch.cat(feature_list, dim=0)
            attention_mask[:, var_idx] = torch.tensor(mask_list, dtype=torch.long, device=device)

        return {
            "features": adapter(clinical_features),
            "attention_mask": attention_mask,
        }
    

    def _maybe_load_vision_pretrain(self):
        ckpt_path = self.config.vision_pretrain
        if not ckpt_path:
            return
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"vision_pretrain not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)

        new_ckpt = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("model.", "")
            new_ckpt[new_key] = v

        msg = self.vision_encoder.load_state_dict(new_ckpt, strict=False)
        print(
            "Loaded vision encoder weights:",
            f"missing={msg.missing_keys}",
            f"unexpected={msg.unexpected_keys}",
        )