# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import PretrainedConfig
from typing import List


class ResNet3DConfig(PretrainedConfig):
    """
    ResNet3D Configuration class for 3D medical imaging.
    
    This is the configuration class to store the configuration of a ResNet3D model.
    It is used to instantiate a ResNet3D model according to the specified arguments,
    defining the model architecture.
    
    Args:
        num_channels (int, optional): Number of input channels. Defaults to 1.
        embedding_size (int, optional): Dimensionality of the embedding layer. Defaults to 64.
        hidden_sizes (List[int], optional): Dimensionality of the hidden layers. Defaults to [64, 128, 256, 512].
        depths (List[int], optional): Number of layers in each block. Defaults to [1, 1, 1, 1].
        layer_type (str, optional): Type of layer to use ('basic' or 'bottleneck'). Defaults to 'basic'.
        hidden_act (str, optional): The non-linear activation function. Defaults to "relu".
        spatial_dims (int, optional): Number of spatial dimensions (2 or 3). Defaults to 3.
        downsample_in_first_stage (bool, optional): Whether to downsample in the first stage. Defaults to False.
        downsample_in_bottleneck (bool, optional): Whether to downsample in bottleneck layers. Defaults to False.
        num_labels (int, optional): Number of classes for classification head. Defaults to 2.
        conv1_kernel_size (int, optional): Kernel size of the first conv layer. Defaults to 7.
        conv1_stride (int, optional): Stride of the first conv layer. Defaults to 2.
        no_max_pool (bool, optional): If True, max pooling after conv1 is not used. Defaults to False.
        widen_factor (float, optional): Width multiplier for the network. Defaults to 1.0.
    """
    
    model_type = "resnet3d"
    
    def __init__(
        self,
        num_channels: int = 1,
        embedding_size: int = 64,
        hidden_sizes: List[int] = None,
        depths: List[int] = None,
        layer_type: str = "basic",
        hidden_act: str = "relu",
        spatial_dims: int = 3,
        downsample_in_first_stage: bool = False,
        downsample_in_bottleneck: bool = False,
        num_labels: int = 2,
        conv1_kernel_size: int = 7,
        conv1_stride: int = 2,
        no_max_pool: bool = False,
        widen_factor: float = 1.0,
        **kwargs,
    ):
        if hidden_sizes is None:
            hidden_sizes = [64, 128, 256, 512]
        
        if depths is None:
            depths = [1, 1, 1, 1]
            
        if spatial_dims not in [2, 3]:
            raise ValueError(f"`spatial_dims` must be 2 or 3, got {spatial_dims}.")
        
        if layer_type not in ["basic", "bottleneck"]:
            raise ValueError(f"`layer_type` must be 'basic' or 'bottleneck', got {layer_type}.")
        
        # Auto mapping for trust_remote_code (must be set before super().__init__)
        if "auto_map" not in kwargs:
            kwargs["auto_map"] = {
                "AutoConfig": "configuration_resnet.ResNet3DConfig",
                "AutoModel": "modeling_resnet.ResNet3DModel",
                "AutoModelForImageClassification": "modeling_resnet.ResNet3DForImageClassification",
                "AutoBackbone": "modeling_resnet.ResNet3DBackbone",
            }
        
        super().__init__(**kwargs)
        
        self.num_channels = num_channels
        self.embedding_size = int(embedding_size * widen_factor)
        self.hidden_sizes = [int(x * widen_factor) for x in hidden_sizes]
        self.depths = depths
        self.layer_type = layer_type
        self.hidden_act = hidden_act
        self.spatial_dims = spatial_dims
        self.downsample_in_first_stage = downsample_in_first_stage
        self.downsample_in_bottleneck = downsample_in_bottleneck
        self.num_labels = num_labels
        self.conv1_kernel_size = conv1_kernel_size
        self.conv1_stride = conv1_stride
        self.no_max_pool = no_max_pool
        self.widen_factor = widen_factor


class ResNet3D10Config(ResNet3DConfig):
    """ResNet3D-10 Configuration."""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('layer_type', 'basic')
        kwargs.setdefault('depths', [1, 1, 1, 1])
        super().__init__(**kwargs)


class ResNet3D18Config(ResNet3DConfig):
    """ResNet3D-18 Configuration."""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('layer_type', 'basic')
        kwargs.setdefault('depths', [2, 2, 2, 2])
        super().__init__(**kwargs)


class ResNet3D34Config(ResNet3DConfig):
    """ResNet3D-34 Configuration."""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('layer_type', 'basic')
        kwargs.setdefault('depths', [3, 4, 6, 3])
        super().__init__(**kwargs)


class ResNet3D50Config(ResNet3DConfig):
    """ResNet3D-50 Configuration."""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('layer_type', 'bottleneck')
        kwargs.setdefault('depths', [3, 4, 6, 3])
        super().__init__(**kwargs)


class ResNet3D101Config(ResNet3DConfig):
    """ResNet3D-101 Configuration."""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('layer_type', 'bottleneck')
        kwargs.setdefault('depths', [3, 4, 23, 3])
        super().__init__(**kwargs)


class ResNet3D152Config(ResNet3DConfig):
    """ResNet3D-152 Configuration."""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('layer_type', 'bottleneck')
        kwargs.setdefault('depths', [3, 8, 36, 3])
        super().__init__(**kwargs)


class ResNet3D200Config(ResNet3DConfig):
    """ResNet3D-200 Configuration."""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('layer_type', 'bottleneck')
        kwargs.setdefault('depths', [3, 24, 36, 3])
        super().__init__(**kwargs)
