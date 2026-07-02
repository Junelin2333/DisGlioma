import torch
import torch.nn as nn

from model.layers.config_resnet import ResNet3DConfig
from model.layers.resnet import ResNet3DModel


class ResNet3DCusConfig(ResNet3DConfig):
    """ResNet3D-10 Configuration."""
    def __init__(self, **kwargs):
        kwargs.setdefault('layer_type', "basic")
        kwargs.setdefault('depths', [3, 4, 6])
        # kwargs.setdefault('depths', [2, 4, 2, 2])
        kwargs.setdefault('hidden_sizes', [64, 128, 256])
        kwargs.setdefault('widen_factor', 1.0)
        kwargs.setdefault('num_channels', 4)
        super().__init__(**kwargs)


class VisionEncoder(nn.Module):
    def __init__(self):
        super(VisionEncoder, self).__init__()

        config = ResNet3DCusConfig()
        self.backbone = ResNet3DModel(config)
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))

    def forward(self, data):
        image = data
        output = self.backbone.forward(image, output_hidden_states=True, return_dict=True)  # type: ignore

        os8 = output['hidden_states'][2]
        h_embed = self.pool(os8).squeeze((-1,-2,-3))

        cls_embed = output['pooler_output'].squeeze((-1,-2,-3))  # [b 256]
        
        return h_embed, cls_embed

