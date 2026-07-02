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

"""PyTorch ResNet3D model for 3D medical imaging."""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn

from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BackboneOutput,
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from transformers.backbone_utils import BackboneMixin

from .config_resnet import ResNet3DConfig


__all__ = [
    "ResNet3DModel",
    "ResNet3D10Model",
    "ResNet3D18Model",
    "ResNet3D34Model",
    "ResNet3D50Model",
    "ResNet3D101Model",
    "ResNet3D152Model",
    "ResNet3D200Model",
    "ResNet3DForImageClassification",
    "ResNet3D10ForImageClassification",
    "ResNet3D18ForImageClassification",
    "ResNet3D34ForImageClassification",
    "ResNet3D50ForImageClassification",
    "ResNet3D101ForImageClassification",
    "ResNet3D152ForImageClassification",
    "ResNet3D200ForImageClassification",
    "ResNet3DBackbone",
    "ResNet3DPreTrainedModel",
]


class ResNet3DConvLayer(nn.Module):
    """
    ResNet3D convolution layer consisting of Conv3d/Conv2d + BatchNorm + Activation.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation: str = "relu",
        spatial_dims: int = 3,
    ):
        super().__init__()
        
        conv_type = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        norm_type = nn.BatchNorm3d if spatial_dims == 3 else nn.BatchNorm2d
        
        self.convolution = conv_type(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.normalization = norm_type(out_channels)
        self.activation = nn.ReLU(inplace=True) if activation == "relu" else nn.Identity()
    
    def forward(self, input: Tensor) -> Tensor:
        hidden_state = self.convolution(input)
        hidden_state = self.normalization(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state


class ResNet3DEmbeddings(nn.Module):
    """
    ResNet3D embeddings (stem) consisting of a single convolution and optional max pooling.
    """
    
    def __init__(self, config: ResNet3DConfig):
        super().__init__()
        self.spatial_dims = config.spatial_dims
        self.num_channels = config.num_channels
        
        self.embedder = ResNet3DConvLayer(
            config.num_channels,
            config.embedding_size,
            kernel_size=config.conv1_kernel_size,
            stride=config.conv1_stride,
            activation=config.hidden_act,
            spatial_dims=config.spatial_dims,
        )
        
        if not config.no_max_pool:
            pool_type = nn.MaxPool3d if config.spatial_dims == 3 else nn.MaxPool2d
            self.pooler = pool_type(kernel_size=3, stride=2, padding=1)
        else:
            self.pooler = None
    
    def forward(self, pixel_values: Tensor) -> Tensor:
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                f"Make sure that the channel dimension of the pixel values match with the one set in the configuration. "
                f"Expected {self.num_channels}, got {num_channels}."
            )
        
        embedding = self.embedder(pixel_values)
        
        if self.pooler is not None:
            embedding = self.pooler(embedding)
        
        return embedding


class ResNet3DShortCut(nn.Module):
    """
    ResNet3D shortcut, used to project the residual features to the correct size.
    If needed, it is also used to downsample the input using stride.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        spatial_dims: int = 3,
    ):
        super().__init__()
        
        conv_type = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        norm_type = nn.BatchNorm3d if spatial_dims == 3 else nn.BatchNorm2d
        
        self.convolution = conv_type(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            bias=False,
        )
        self.normalization = norm_type(out_channels)
    
    def forward(self, input: Tensor) -> Tensor:
        hidden_state = self.convolution(input)
        hidden_state = self.normalization(hidden_state)
        return hidden_state


class ResNet3DBasicLayer(nn.Module):
    """
    A classic ResNet's residual layer composed by two 3x3(x3) convolutions.
    Used in ResNet-10, 18, 34.
    """
    
    expansion: int = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation: str = "relu",
        spatial_dims: int = 3,
    ):
        super().__init__()
        
        should_apply_shortcut = in_channels != out_channels or stride != 1
        
        self.shortcut = (
            ResNet3DShortCut(in_channels, out_channels, stride=stride, spatial_dims=spatial_dims)
            if should_apply_shortcut
            else nn.Identity()
        )
        
        self.layer = nn.Sequential(
            ResNet3DConvLayer(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                activation=activation,
                spatial_dims=spatial_dims,
            ),
            ResNet3DConvLayer(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                activation=None,
                spatial_dims=spatial_dims,
            ),
        )
        
        self.activation = nn.ReLU(inplace=True) if activation == "relu" else nn.Identity()
    
    def forward(self, hidden_state: Tensor) -> Tensor:
        residual = hidden_state
        hidden_state = self.layer(hidden_state)
        residual = self.shortcut(residual)
        hidden_state += residual
        hidden_state = self.activation(hidden_state)
        return hidden_state


class ResNet3DBottleNeckLayer(nn.Module):
    """
    A classic ResNet's bottleneck layer composed by three convolutions.
    Used in ResNet-50, 101, 152, 200.
    
    The first 1x1 convolution reduces the input by a factor of 4.
    The second 3x3 convolution is the main computation.
    The last 1x1 convolution expands back to out_channels.
    """
    
    expansion: int = 4
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation: str = "relu",
        spatial_dims: int = 3,
        downsample_in_bottleneck: bool = False,
    ):
        super().__init__()
        
        should_apply_shortcut = in_channels != out_channels * self.expansion or stride != 1
        reduces_channels = out_channels
        
        self.shortcut = (
            ResNet3DShortCut(
                in_channels,
                out_channels * self.expansion,
                stride=stride,
                spatial_dims=spatial_dims,
            )
            if should_apply_shortcut
            else nn.Identity()
        )
        
        self.layer = nn.Sequential(
            ResNet3DConvLayer(
                in_channels,
                reduces_channels,
                kernel_size=1,
                stride=stride if downsample_in_bottleneck else 1,
                activation=activation,
                spatial_dims=spatial_dims,
            ),
            ResNet3DConvLayer(
                reduces_channels,
                reduces_channels,
                kernel_size=3,
                stride=stride if not downsample_in_bottleneck else 1,
                activation=activation,
                spatial_dims=spatial_dims,
            ),
            ResNet3DConvLayer(
                reduces_channels,
                out_channels * self.expansion,
                kernel_size=1,
                stride=1,
                activation=None,
                spatial_dims=spatial_dims,
            ),
        )
        
        self.activation = nn.ReLU(inplace=True) if activation == "relu" else nn.Identity()
    
    def forward(self, hidden_state: Tensor) -> Tensor:
        residual = hidden_state
        hidden_state = self.layer(hidden_state)
        residual = self.shortcut(residual)
        hidden_state += residual
        hidden_state = self.activation(hidden_state)
        return hidden_state


class ResNet3DStage(nn.Module):
    """
    A ResNet3D stage composed of stacked layers.
    """
    
    def __init__(
        self,
        config: ResNet3DConfig,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        depth: int = 2,
    ):
        super().__init__()
        
        layer_class = (
            ResNet3DBottleNeckLayer
            if config.layer_type == "bottleneck"
            else ResNet3DBasicLayer
        )
        
        # Determine expansion factor
        expansion = layer_class.expansion
        
        if config.layer_type == "bottleneck":
            first_layer = layer_class(
                in_channels,
                out_channels,
                stride=stride,
                activation=config.hidden_act,
                spatial_dims=config.spatial_dims,
                downsample_in_bottleneck=config.downsample_in_bottleneck,
            )
        else:
            first_layer = layer_class(
                in_channels,
                out_channels,
                stride=stride,
                activation=config.hidden_act,
                spatial_dims=config.spatial_dims,
            )
        
        # Subsequent layers
        subsequent_layers = [
            layer_class(
                out_channels * expansion,
                out_channels,
                stride=1,
                activation=config.hidden_act,
                spatial_dims=config.spatial_dims,
            )
            for _ in range(depth - 1)
        ]
        
        self.layers = nn.Sequential(first_layer, *subsequent_layers)
    
    def forward(self, input: Tensor) -> Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class ResNet3DEncoder(nn.Module):
    """
    ResNet3D encoder consisting of multiple stages.
    """
    
    def __init__(self, config: ResNet3DConfig):
        super().__init__()
        self.stages = nn.ModuleList([])
        
        # First stage
        self.stages.append(
            ResNet3DStage(
                config,
                config.embedding_size,
                config.hidden_sizes[0],
                stride=2 if config.downsample_in_first_stage else 1,
                depth=config.depths[0],
            )
        )
        
        # Determine expansion factor
        layer_class = (
            ResNet3DBottleNeckLayer
            if config.layer_type == "bottleneck"
            else ResNet3DBasicLayer
        )
        expansion = layer_class.expansion
        
        # Subsequent stages
        for i in range(1, len(config.hidden_sizes)):
            self.stages.append(
                ResNet3DStage(
                    config,
                    config.hidden_sizes[i - 1] * expansion,
                    config.hidden_sizes[i],
                    stride=2,
                    depth=config.depths[i],
                )
            )
    
    def forward(
        self,
        hidden_state: Tensor,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        hidden_states = () if output_hidden_states else None
        
        for stage_module in self.stages:
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state,)
            
            hidden_state = stage_module(hidden_state)
        
        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)
        
        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states] if v is not None)
        
        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_state,
            hidden_states=hidden_states,
        )


class ResNet3DPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    
    config_class = ResNet3DConfig
    base_model_prefix = "resnet3d"
    main_input_name = "pixel_values"
    _no_split_modules = ["ResNet3DBasicLayer", "ResNet3DBottleNeckLayer"]
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


class ResNet3DModel(ResNet3DPreTrainedModel):
    """
    ResNet3D model for feature extraction.
    
    This model outputs the hidden states and pooled output from the last residual block.
    Based on: `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`_.
    Adapted for 3D medical imaging with Hugging Face Hub compatibility.
    """
    
    def __init__(self, config: ResNet3DConfig):
        super().__init__(config)
        self.config = config
        
        self.embedder = ResNet3DEmbeddings(config)
        self.encoder = ResNet3DEncoder(config)
        
        pool_type = nn.AdaptiveAvgPool3d if config.spatial_dims == 3 else nn.AdaptiveAvgPool2d
        self.pooler = pool_type((1, 1, 1) if config.spatial_dims == 3 else (1, 1))
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        pixel_values: Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndNoAttention]:
        """
        Forward pass returning hidden states and pooled output.
        
        Args:
            pixel_values (Tensor): Input tensor of shape (batch_size, channels, *spatial_dims)
            output_hidden_states (bool, optional): Whether to return hidden states from all layers
            return_dict (bool, optional): Whether to return a ModelOutput instead of a tuple
            
        Returns:
            BaseModelOutputWithPoolingAndNoAttention or tuple: Model outputs
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        embedding_output = self.embedder(pixel_values)
        
        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        last_hidden_state = encoder_outputs[0]
        
        pooled_output = self.pooler(last_hidden_state)
        
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        
        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


class ResNet3DForImageClassification(ResNet3DPreTrainedModel):
    """
    ResNet3D model with an image classification head on top.
    
    This model outputs logits and optionally computes the loss if labels are provided.
    """
    
    def __init__(self, config: ResNet3DConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        self.resnet3d = ResNet3DModel(config)
        
        # Classification head
        layer_class = (
            ResNet3DBottleNeckLayer
            if config.layer_type == "bottleneck"
            else ResNet3DBasicLayer
        )
        expansion = layer_class.expansion
        final_hidden_size = config.hidden_sizes[-1] * expansion
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity(),
        )
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:
        """
        Forward pass with optional loss computation.
        
        Args:
            pixel_values (Tensor): Input tensor of shape (batch_size, channels, *spatial_dims)
            labels (Tensor, optional): Labels for computing the classification loss
            output_hidden_states (bool, optional): Whether to return hidden states from all layers
            return_dict (bool, optional): Whether to return a ModelOutput instead of a tuple
            
        Returns:
            ImageClassifierOutputWithNoAttention or tuple: Model outputs with logits and optional loss
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.resnet3d(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        pooled_output = outputs.pooler_output if return_dict else outputs[1]
        
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                # Regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                # Classification
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


class ResNet3DBackbone(ResNet3DPreTrainedModel, BackboneMixin):
    """
    ResNet3D backbone, to be used with frameworks like DETR and MaskFormer.
    """
    
    has_attentions = False
    
    def __init__(self, config: ResNet3DConfig):
        super().__init__(config)
        super()._init_backbone(config)
        
        layer_class = (
            ResNet3DBottleNeckLayer
            if config.layer_type == "bottleneck"
            else ResNet3DBasicLayer
        )
        expansion = layer_class.expansion
        
        self.num_features = [config.embedding_size] + [
            size * expansion for size in config.hidden_sizes
        ]
        
        self.embedder = ResNet3DEmbeddings(config)
        self.encoder = ResNet3DEncoder(config)
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        pixel_values: Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BackboneOutput]:
        """
        Forward pass for backbone.
        
        Returns:
            BackboneOutput: Backbone outputs with feature maps
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        embedding_output = self.embedder(pixel_values)
        
        outputs = self.encoder(embedding_output, output_hidden_states=True, return_dict=True)
        
        hidden_states = outputs.hidden_states
        
        feature_maps = ()
        for idx, stage in enumerate(self.stage_names):
            if stage in self.out_features:
                feature_maps += (hidden_states[idx],)
        
        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output += (outputs.hidden_states,)
            return output
        
        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=None,
        )




