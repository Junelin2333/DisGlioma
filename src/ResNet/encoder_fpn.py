import torch
import torch.nn as nn

from model.layers.config_resnet import ResNet3DConfig
from model.layers.resnet import ResNet3DModel


class ResNet3DCusConfig(ResNet3DConfig):
    """ResNet3D-10 Configuration."""
    def __init__(self, **kwargs):
        kwargs.setdefault('layer_type', "bottleneck")
        kwargs.setdefault('depths', [3, 4, 6, 3])
        # kwargs.setdefault('depths', [2, 4, 2, 2])
        kwargs.setdefault('hidden_sizes', [64, 128, 256, 512])
        kwargs.setdefault('widen_factor', 1.0)
        kwargs.setdefault('num_channels', 4)
        super().__init__(**kwargs)


class VisionEncoder(nn.Module):
    def __init__(self):
        super(VisionEncoder, self).__init__()

        config = ResNet3DCusConfig()
        self.backbone = ResNet3DModel(config)
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))

        self.proj1 = nn.Sequential(
            nn.Linear(256, 768),
            nn.GELU(),
        )
        self.proj2 = nn.Sequential(
            nn.Linear(512, 768),
            nn.GELU(),
        )

        self.proj3 = nn.Sequential(
            nn.Linear(1024, 768),
            nn.GELU(),
        )
        self.proj4 = nn.Sequential(
            nn.Linear(2048, 768),
            nn.GELU(),
        )

        self.hazard_proj = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
        )

    def forward(self, data):
        image = data
        output = self.backbone.forward(image, output_hidden_states=True, return_dict=True)  # type: ignore

        os8 = output['hidden_states'][2]
        h_embed = self.pool(os8).squeeze((-1,-2,-3))
        h_embed = self.hazard_proj(h_embed)
        cls_embed = self.fpn_features(output['hidden_states'])

        return h_embed, cls_embed

    def fpn_features(self, features):
        os4 = features[-4].mean(dim=(-1,-2,-3))
        os8 = features[-3].mean(dim=(-1,-2,-3))
        os16 = features[-2].mean(dim=(-1,-2,-3))
        os32 = features[-1].mean(dim=(-1,-2,-3))

        os4 = self.proj1(os4)
        os8 = self.proj2(os8)
        os16 = self.proj3(os16)
        os32 = self.proj4(os32)

        features = torch.stack([os4, os8, os16, os32], dim=1)

        return features