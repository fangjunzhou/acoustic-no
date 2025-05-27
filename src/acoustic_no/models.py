import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO

class UNet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Encoder path
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Decoder path
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(channels*2, channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.deconv0 = nn.Sequential(
            nn.ConvTranspose2d(channels*2, channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.output_layer = nn.Conv2d(channels*2, channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        conv2 = self.conv2_1(self.conv2(conv1))
        conv3 = self.conv3_1(self.conv3(conv2))
        
        # Decoder with skip connections
        deconv2 = self.deconv2(conv3)
        concat2 = torch.cat([conv2, deconv2], dim=1)
        deconv1 = self.deconv1(concat2)
        concat1 = torch.cat([conv1, deconv1], dim=1)
        deconv0 = self.deconv0(concat1)
        concat0 = torch.cat([x, deconv0], dim=1)
        
        return self.output_layer(concat0)

class UFNOBlock(nn.Module):
    def __init__(self, n_modes, width, has_unet=False):
        super().__init__()
        self.has_unet = has_unet
        
        # Fourier path
        self.fno = FNO(
            n_modes=n_modes,
            in_channels=width,
            out_channels=width,
            hidden_channels=width
        )
        
        # Spatial path
        self.w = nn.Conv2d(width, width, 1)
        
        # Optional U-Net path
        if has_unet:
            self.unet = UNet(width)
            
    def forward(self, x):
        # Fourier path
        x1 = self.fno(x)
        
        # Spatial path
        x2 = self.w(x)
        
        if self.has_unet:
            # U-Net path
            x3 = self.unet(x)
            x = x1 + x2 + x3
        else:
            x = x1 + x2
            
        return F.relu(x)

class UFNO(nn.Module):
    def __init__(
        self,
        n_modes=(16, 16),
        in_channels=193,  # DEPTH * 3 + 1 for pressure, velocity (x,y), and alpha
        out_channels=64,  # DEPTH for pressure prediction
        width=64,
        n_layers=6,
    ):
        super().__init__()
        
        self.fc0 = nn.Linear(in_channels, width)
        
        # UFNO blocks
        self.blocks = nn.ModuleList([
            UFNOBlock(
                n_modes=n_modes,
                width=width,
                has_unet=(i >= 3)  # Only last 3 blocks have U-Net
            ) for i in range(n_layers)
        ])
        
        # Output layers
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)
        
    def forward(self, x):
        # Input projection
        x = self.fc0(x.permute(0, 2, 3, 1))
        x = x.permute(0, 3, 1, 2)
        
        # Process through UFNO blocks
        for block in self.blocks:
            x = block(x)
            
        # Output projection
        x = x.permute(0, 2, 3, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        
        return x 