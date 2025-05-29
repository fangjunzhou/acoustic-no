import torch
from torch import nn
from torch.nn.init import xavier_uniform_

from acoustic_no.cno.cno_layers import *

class CNOModel(nn.Module):
    def __init__(self, input_channels, hidden_channels: list[int], layer_sizes: list[int], output_channels):
        super(CNOModel, self).__init__()
        assert len(hidden_channels) == len(layer_sizes), "hidden_channels and layer_sizes must have the same length"
        self.lifting = nn.Conv2d(input_channels, hidden_channels[0], kernel_size=1, stride=1, padding=0)
        self.downsample_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        self.cno_layers = nn.ModuleList()
        self.identity_blocks = nn.ModuleList()
        for i in range(len(hidden_channels) - 1):
            self.cno_layers.append(CNOLayer(hidden_channels[i], hidden_channels[i], layer_sizes[i]))
            self.downsample_blocks.append(DownsampleBlock(hidden_channels[i], hidden_channels[i + 1]))
            self.identity_blocks.append(IdentityBlock(hidden_channels[i] * 2, hidden_channels[i]))
            self.upsample_blocks.append(UpsampleBlock(hidden_channels[i + 1], hidden_channels[i]))
        self.cno_layers.append(CNOLayer(hidden_channels[-1], hidden_channels[-1], layer_sizes[-1]))
        self.projection = nn.Conv2d(hidden_channels[0], output_channels, kernel_size=1, stride=1, padding=0)
        self.num_layers = len(hidden_channels)

        # Initialize weights
        xavier_uniform_(self.lifting.weight)
        xavier_uniform_(self.projection.weight)

    def _foward(self, x, depth: int):
        if depth == self.num_layers - 1:
            return self.cno_layers[depth](x)
        else:
            x_down = self.downsample_blocks[depth](x)
            x_recursive = self._foward(x_down, depth + 1)
            x_up = self.upsample_blocks[depth](x_recursive)
            x_layer = self.cno_layers[depth](x)
            x_identity = self.identity_blocks[depth](torch.cat((x_layer, x_up), dim=1))
            return x_identity

    def forward(self, x):
        # Lift input to hidden space
        x = self.lifting(x)
        # Forward through CNO layers
        x = self._foward(x, 0)
        # Project to output space
        x = self.projection(x)
        return x