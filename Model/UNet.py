import torch
import torch.nn as nn, Tensor
import torch.nn.functional as F
from typing import Optional, Union, List, Dict, Callable

class DoubleConv(nn.Module):
    """
    Architecture:
        [Conv2D 3x3 -> BatchNorm -> ReLU] * 2
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
    ) -> None:
        
        super(DoubleConv, self).__init__()
        
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            # Block 1
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            
            # Block 2
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x: Tensor) -> Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """
    Architecture:
        MaxPool2D 2x2: downsample by stride 2 
        -> DoubleConv: keeping the same number of channels but increase # features
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )
        
    def forward(self, x: Tensor) -> Tensor:
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
    Architecture:
        -Input
        ->NewSample = Upsample
        -Concatenate
           +[Input, NewSample]
        ->DoubleConv
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True, 
    ) -> None:
        """
        Args:
            bilinear (bool, optional): Defaults to True.
                When we a pixel attempting to upsample, and find the value by interpolation around the existing pixels
                
                        d1             d2
                |----------------|------------|
                V1              q1            V2
                O________________O____________O __
                |                |            |  |   
                |     A4     (q) |    A3      |  |  d3   
                |________________0____________| _|_ 
                |     A2         |    A1      |  |   
                O________________O____________O _|_ d4
                V3              q2            V4
                
                q1 = V1*d2 + V2*d1
                q2 = V3*d4 + V4*d3
                q = q1*d3 + q2*d4
                -> q = V1*A1 + V2*A2 + V3*A3 + V4*A4
        """
        super(Up, self).__init__()
        
        # Check bilinear, use normal convolution to reduce # features
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, # H_out = H_in * scale_factor, W_out = W_in * scale_factor
                mode='bilinear',
                align_corners=True,  # preserve the values at the corners
            )
            
            self.conv = DoubleConv(
                in_channels=in_channels,
                out_channels=out_channels,
                mid_channels=in_channels // 2,
            )

        else:
            # Upsampling followed by a normal conv we multiply number and sum of these pixels
            self.up = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels // 2, # Integer value size of the output
                kernel_size=2,
                stride=2,
            )
            
            self.conv = DoubleConv(
                in_channels=in_channels,
                out_channels=out_channels,
            )
    
    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        
        x1 = self.up(x1)
        
        # Input is NxCHW
        diffY = x2.size()[2] - x1.size()[2] # Height layer
        diffX = x2.size()[3] - x1.size()[3] # Width layer
        
        x1 = F.pad(
            input=x1,
            pad=[
                diffX // 2, diffX - diffX // 2,
                diffY // 2, diffY - diffY // 2,
            ]
        )
        
        # Concatenate along the channels axis
        # x1: New upsampled layer
        # x2: Sample same level in downsample path
        # Skip connection technique
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
        

class OutConv(nn.Module):
    """
    Architecture:
        Conv2D 1x1: reduce # features to # classes
    """
    
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
    ) -> None:
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)
    


class UNet(nn.Module):
    
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        bilinear: bool = False,
    ) -> None:
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inc = (DoubleConv(n_channels, 64))
        
        # Define 4 downsample layers
        self.down1 = (
            Down(
                in_channels=64,
                out_channels=128,
            )
        )
        self.down2 = (
            Down(
                in_channels=128,
                out_channels=256,
            )
        )
        self.down3 = (
            Down(
                in_channels=256,
                out_channels=512,
            )
        )
        
        factor = 2 if bilinear else 1
        
        self.down4 = (
            Down(
                in_channels=512,
                out_channels=1024 // factor,
            )
        )
        
        # Define 4 upsample layers
        self.up1 = (
            Up(
                in_channels=1024,
                out_channels=512 // factor,
                bilinear=bilinear,
            )
        )
        self.up2 = (
            Up(
                in_channels=512,
                out_channels=256 // factor,
                bilinear=bilinear,
            )
        )
        self.up3 = (
            Up(
                in_channels=256,
                out_channels=128 // factor,
                bilinear=bilinear,
            )
        )
        self.up4 = (
            Up(
                in_channels=128,
                out_channels=64,
                bilinear=bilinear,
            )
        )
        
        # Output layer
        self.outc = OutConv(
            in_channels=64,
            out_channels=n_classes,
        )
        
    
    def forward(self, x: Tensor) -> Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2) 
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return logits
    
    def use_checkpoint(self) -> None:
        
        # Store the checkpoint all layer 
        
        # Input process
        self.inc = torch.utils.checkpoint.checkpoint(self.inc)
        
        # Encoder: Downsample phase
        self.down1 = torch.utils.checkpoint.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint.checkpoint(self.down4)
        
        # Decoder: Upsample phase
        self.up1 = torch.utils.checkpoint.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint.checkpoint(self.up4)
        
        # Output process
        self.outc = torch.utils.checkpoint.checkpoint(self.outc) 