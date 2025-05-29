import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

class ActivationLayer(nn.Module):
    def __init__(self, activation_type='relu'):
        super(ActivationLayer, self).__init__()
        if activation_type == 'relu':
            self.activation = nn.ReLU()
        elif activation_type == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_type == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")

    def forward(self, x):
        # Upsample the input tensor into a higher bandlimit.
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # Apply the activation function.
        x = self.activation(x)
        # Downsample the tensor back to the original size.
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        return x
    
class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.activation = ActivationLayer('relu')

        # Initialize weights using Xavier uniform initialization
        xavier_uniform_(self.conv.weight)

    def forward(self, x):
        # Apply convolution
        x = self.conv(x)
        # Apply activation
        x = self.activation(x)
        # Downsample the tensor by a factor of 2
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.activation = ActivationLayer('relu')

        # Initialize weights using Xavier uniform initialization
        xavier_uniform_(self.conv.weight)

    def forward(self, x):
        # Apply convolution
        x = self.conv(x)
        # Apply activation
        x = self.activation(x)
        # Upsample the tensor by a factor of 2
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return x
    
class IdentityBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IdentityBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.activation = ActivationLayer('relu')

        # Initialize weights using Xavier uniform initialization
        xavier_uniform_(self.conv.weight)

    def forward(self, x):
        # Apply convolution
        x = self.conv(x)
        # Apply activation
        x = self.activation(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.activation = ActivationLayer('relu')

        # Initialize weights using Xavier uniform initialization
        xavier_uniform_(self.conv1.weight)
        xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        # Apply first convolution and activation
        residual = x
        x = self.conv1(x)
        x = self.activation(x)
        # Apply second convolution
        x = self.conv2(x)
        # Add the residual connection
        x += residual
        return self.activation(x)  # Apply activation after adding residual

class CNOLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_residual_blocks=2):
        super(CNOLayer, self).__init__()
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(in_channels) for _ in range(num_residual_blocks)]
        )
        self.identity_block = IdentityBlock(in_channels, out_channels)
    
    def forward(self, x):
        # Pass through residual blocks
        for block in self.residual_blocks:
            x = block(x)
        # Pass through identity block
        x = self.identity_block(x)
        return x