# Create model inception v3
import warnings
from collections import namedtuple
from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor


__all__ = [
    "Inception3",
    "InceptionOutputs",
    "_InceptionOutputs",
]

# Declare the output of inception model
InceptionOutputs = namedtuple(
    typename='InceptionOutputs',
    field_names=['logits', 'aux_logits']    
)

InceptionOutputs.__annotations__ = {
    'logits': Tensor,
    'aux_logits': Optional[Tensor] # Optional: Tensor or None
}

# Declare backward
_InceptionOutputs = InceptionOutputs


# Define Basic Conv2d

class BasicConv2d(nn.Module):
    """
    Architecture: Conv2D -> BatchNorm2D -> ReLU
    """
    def __init__(self, in_channels: int,
                 out_channels: int,
                 **kwargs: Any) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=False,
            **kwargs
        )
        self.bn = nn.BatchNorm2d(
            num_features=out_channels,
            eps=0.001
        )            

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

# Define InceptionA
class InceptionA(nn.Module):
    """
    Architecture:
        -Concatenate:
            +Path 1 (1x1): Conv2d 1x1x3
            +Path 2 (5x5): Conv2d 1x1x3 -> Conv2d 5x5x48
            +Path 3 (3x3): Conv2d 1x1x3 -> Conv2d 3x3x64 -> Conv2d 3x3x96
            +Path 4 (Average pooling): AvgPool2d 3x3x1 -> Conv2d 1x1x32
    """

    def __init__(self, in_channels: int,
                 pool_features: int,
                 conv_block: Optional[Callable[..., nn.Module]] = None) -> None:
        """_summary_

        Args:
            conv_block (Optional[Callable[..., nn.Module]]): Any type to nn.Module pytorch or None Conv Block is deafault
        """
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        
        # Path 1
        self.branch1x1 = conv_block(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=1
        )
        
        # Path 2
        self.branch5x5_1 = conv_block(
            in_channels=in_channels,
            out_channels=48,
            kernel_size=1
        )
        self.branch5x5_2 = conv_block(
            in_channels=48,
            out_channels=64,
            kernel_size=5,
            padding=2
        )
        
        # Path 3
        self.branch3x3dbl_1 = conv_block(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=1
        )

        self.branch3x3dbl_2 = conv_block(
            in_channels=64,
            out_channels=96,
            kernel_size=3,
            padding=1
        )
        
        self.branch3x3dbl_3 = conv_block(
            in_channels=96,
            out_channels=96,
            kernel_size=3,
            padding=1
        )
        
        # Path 4
        self.branch_pool = conv_block(
            in_channels=in_channels,
            out_channels=pool_features,
            kernel_size=1
        )
    
    def _forward(self, x: Tensor) -> List[Tensor]:
        # Path 1
        branch1x1 = self.branch1x1(x)

        # Path 2
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        
        # Path 3
        branch3x3 = self.branch3x3dbl_1(x)
        branch3x3 = self.branch3x3dbl_2(branch3x3)
        branch3x3 = self.branch3x3dbl_3(branch3x3)
        
        # Path 4
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        
        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        
        return outputs
    
    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        
        # Concatenate by expanding depth of image
        return torch.cat(outputs, 1)
        
# Define InceptionB
class InceptionB(nn.Module):
    """
    Architecture:
        -Concatenate:
            +Path 1 (3x3): Conv2d 3x3x3
            +Path 2 (3x3): Conv2d 1x1x3 -> Conv2d 3x3x64 -> Conv2d 3x3x96
            +Path 3 (Max pooling): AvgPool2d 3x3x1
    """
    
    
    def __init__(
        self, 
        in_channels: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        
        super(InceptionB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        
        # Path 1
        self.branch3x3 = conv_block(
            in_channels=in_channels,
            out_channels=384,
            kernel_size=3,
            stride=2
        )
        
        # Path 2
        self.branch3x3dbl_1 = conv_block(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=1
        
        )
        
        self.branch3x3dbl_2 = conv_block(
            in_channels=64,
            out_channels=96,
            kernel_size=3,
            padding=1
        )
        
        self.branch3x3dbl_3 = conv_block(
            in_channels=96,
            out_channels=96,
            kernel_size=3,
            stride=2
        )
        
    
    def _forward(self, x: Tensor) -> List[Tensor]:
        # Path 1
        brach3x3 = self.branch3x3(x)
        
        # Path 2
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        
        # Path 3
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        
        outputs = [brach3x3, branch3x3dbl, branch_pool]
        
        return outputs
    
    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        
        # Concatenate by expanding depth of image
        return torch.cat(outputs, 1)
        

# Define InceptionC
class InceptionC(nn.Module):
    """
    Architecture:
        -Concatenate:
            +Path 1 (1x1): Conv2d 1x1x3
            
            +Path 2 (7x7): 
                Conv2d 1x1x3 -> 
                
                Conv2d 1x7x(# filter 7x7) -> 
                Conv2d 7x1x(# filter 7x7)
                
            +Path 3 (7x7): 
                Conv2d 1x1x3 -> 
                
                Conv2d 7x1x(# filter 7x7) -> 
                Conv2d 1x7x(# filter 7x7) -> 
                
                Conv2d 7x1x(# filter 7x7) -> 
                Conv2d 1x7x(# filter 7x7)
                
            +Path 4 (Average pooling): 
                AvgPool2d 3x3x1 -> 
                Conv2d 1x1x3
    """
    def __init__(
        self, 
        in_channels: int,
        channels_7x7: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        
        super(InceptionC, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        
        # Path 1
        self.branch1x1 = conv_block(
            in_channels=in_channels,
            out_channels=192,
            kernel_size=1
        )
        
        # Path 2
        num7 = channels_7x7
        self.branch7x7_1 = conv_block(
            in_channels=in_channels,
            out_channels=num7,
            kernel_size=1
        )
        
        self.branch7x7_2 = conv_block(
            in_channels=num7,
            out_channels=num7,
            kernel_size=(1, 7),
            padding=(0, 3)
        )
        
        self.branch7x7_3 = conv_block(
            in_channels=num7,
            out_channels=192,
            kernel_size=(7, 1),
            padding=(3, 0)
        )
        
        # Path 3
        self.branch7x7dbl_1 = conv_block(
            in_channels=in_channels,
            out_channels=num7,
            kernel_size=1
        )
        
        self.branch7x7dbl_2 = conv_block(
            in_channels=num7,
            out_channels=num7,
            kernel_size=(7, 1),
            padding=(3, 0)
        )
        
        self.branch7x7dbl_3 = conv_block(
            in_channels=num7,
            out_channels=num7,
            kernel_size=(1, 7),
            padding=(0, 3)
        )
        
        self.branch7x7dbl_4 = conv_block(
            in_channels=num7,
            out_channels=num7,
            kernel_size=(7, 1),
            padding=(3, 0)
        )
        
        self.branch7x7dbl_5 = conv_block(
            in_channels=num7,
            out_channels=192,
            kernel_size=(1, 7),
            padding=(0, 3)
        )
        
        
        # Path 4
        self.branch_pool = conv_block(
            in_channels=in_channels,
            out_channels=192,
            kernel_size=1
        )
        
    
    def _forward(self, x: Tensor) -> List[Tensor]:
        # Path 1
        branch1x1 = self.branch1x1(x)
        
        # Path 2
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        
        # Path 3
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        
        # Path 4
        branch_pool = F.avg_pool2d(x, 
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        branch_pool = self.branch_pool(branch_pool)
        
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        
        # Concatenate by expanding depth of image
        return torch.cat(outputs, 1)

# Define InceptionD
class InceptionD(nn.Module):
    """
    Architecture:
        -Concatenate:
            +Path 1 (1x1): Conv2d 1x1x3 -> Conv2d 3x3x192
            
            +Path 2 (7x7): 
                Conv2d 1x1x3 -> 
                Conv2d 1x7x192 -> 
                Conv2d 7x1x192 ->
                Conv2d 3x3x192
            
            +Path 3 (Max pooling): 
                MaxPool2d 3x3x1
    """
    def __init__(
        self,
        in_channels: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        
        super(InceptionD, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
            
        # Path 1
        self.branch3x3_1 = conv_block(
            in_channels=in_channels,
            out_channels=192,
            kernel_size=1
        )
        
        self.branch3x3_2 = conv_block(
            in_channels=192,
            out_channels=320,
            kernel_size=3,
            stride=2
        )
        
        # Path 2
        self.branch7x7x3_1 = conv_block(
            in_channels=in_channels,
            out_channels=192,
            kernel_size=1
        )
        
        self.branch7x7x3_2 = conv_block(
            in_channels=192,
            out_channels=192,
            kernel_size=(1, 7),
            padding=(0, 3)
        )
        
        self.branch7x7x3_3 = conv_block(
            in_channels=192,
            out_channels=192,
            kernel_size=(7, 1),
            padding=(3, 0)
        )

        self.branch3x3_4 = conv_block(
            in_channels=192,
            out_channels=192,
            kernel_size=3,
            stride=2
        )
    
    def _forward(self, x: Tensor) -> List[Tensor]:
        
        # Path 1
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        
        # Path 2
        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch3x3_4(branch7x7x3)
        
        # Path 3
        branch_pool = F.max_pool2d(x,
                                   kernel_size=3,
                                   stride=2)
        
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> None:
        outputs = self._forward(x)
        
        # Concatenate by expanding depth of image
        return torch.cat(outputs, 1)
    
# Define InceptionE
class InceptionE(nn.Module):
    """
    Architecture:
        -Concatenate:
            +Path 1 (1x1): Conv2d 1x1x3
            
            +Path 2 (3x3): 
                Conv2d 1x1x3 -> 
                Concatenate:
                    +Conv2d 1x3x384
                    +Conv2d 3x1x384
            
            +Path 3 (3x3):
                Conv2d 1x1x3 ->
                Conv2d 3x3x448 ->
                Concatenate:
                    +Conv2d 1x3x384
                    +Conv2d 3x1x384
            
            +Path 3 (Average pooling): 
                AvgPool2d 3x3x1 ->
                Conv2d 1x1x3
    """
    
    def __init__(
        self, 
        in_channels: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        
        super(InceptionE, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        # Path 1 
        self.branch1x1 = conv_block(
            in_channels=in_channels,
            out_channels=320,
            kernel_size=1
        )
        
        # Path 2
        self.branch3x3_1 = conv_block(
            in_channels=in_channels,
            out_channels=384,
            kernel_size=1
        )
        
        self.branch3x3_2a = conv_block(
            in_channels=384,
            out_channels=384,
            kernel_size=(1, 3),
            padding=(0, 1)
        )
        
        self.branch3x3_2b = conv_block(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 1),
            padding=(1, 0)
        )
        
        # Path 3
        self.branch3x3dbl_1 = conv_block(
            in_channels=in_channels,
            out_channels=448,
            kernel_size=1
        )
        
        self.branch3x3_2 = conv_block(
            in_channels=448,
            out_channels=384,
            kernel_size=3,
            padding=1
        )
        
        self.branch3x3_3a = conv_block(          
            in_channels=384,
            out_channels=384,
            kernel_size=(1, 3),
            padding=(0, 1)
        ) 
        self.branch3x3_3b = conv_block(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 1),
            padding=(1, 0)
        )
        
        # Path 4
        self.branch_pool = conv_block(
            in_channels=in_channels,
            out_channels=192,
            kernel_size=1
        )
    
    def _forward(self, x:Tensor) -> List[Tensor]:
        
        # Path 1
        branch1x1 = self.branch1x1(x)

        # Path 2
        branch3x3 = self.branch3x3_1(x)
        
        ## Concatenate 2a and 2b into one
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3)
        ]
        
        branch3x3 = torch.cat(branch3x3, 1)
        
        # Path 3
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3_3a(branch3x3dbl),
            self.branch3x3_3b(branch3x3dbl)
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        
        branch_pool = F.avg_pool2d(x,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        branch_pool = self.branch_pool(branch_pool)
        
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        
        return outputs
    
    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        
        # Concatenate by expanding depth of image
        return torch.cat(outputs, 1)
    
class InceptionAux(nn.Module):
    """
    Architecture:
        +AvgPool2d 5x5x3 ->
        +Conv2d 1x1x3
        +Conv2d 5x5x768
        +Linear 768x2
    """
    def __init__(self, 
                 in_channels: int,
                 num_classes: int,
                 conv_block: Optional[Callable[..., nn.Module]] = None) -> None:
        
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        
        self.conv0 = conv_block(
            in_channels=in_channels,
            out_channels=128,
            kernel_size=1
        )
        
        self.conv1 = conv_block(
            in_channels=128,
            out_channels=768,
            kernel_size=5
        )
        
        # type: ignore assign
        self.conv1.stddev = 0.01
    
        self.fc = nn.Linear(
            in_features=768,
            out_features=num_classes
        )
        
        # type: ignore assign
        self.fc.stddev = 0.001
    
    def forward(self, x: Tensor) -> Tensor:
        #Input: N x 768 x 17 x 17
        x = F.avg_pool2d(x, 
                         kernel_size=5,
                         stride=3)
        
        #Output: N x 768 x 5 x 5
        x = self.conv0(x)

        #Output: N x 128 x 5 x 5
        x = self.conv1(x)
        
        #Output: N x 768 x 1 x 1
        x = F.adaptive_avg_pool2d(x, (1, 1))
        
        #Output: N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        
        #Output: N x 768
        x = self.fc(x)
        
        #Output: N x 2
        return x

class InceptionV3(nn.Module):
    """
    Architecture:
        Concatenate:
            +Path 1:
                -Conv2d 3x3x3
                ->Conv2d 3x3x32
                ->Conv2d 3x3x32 + 1 padding
                ->MaxPool2d 3x3x1
                ->Conv2d 1x1x64
                ->Conv2d 3x3x80
                ->MaxPool2d 3x3x1
                ->InceptionA x 3: size 32, 64, 64 
                ->InceptionB
                ->InceptionC x 4: size 128, 160, 160, 192
                ->InceptionD
                ->InceptionE x 2: size 1280, 2048
                ->AvgPool2d 1x1
                ->Dropout 0.5
                ->Linear 2048x2
            +Path 2:
                -InceptionAux
    """
    def __init__(
        self,
        num_classes: int = 2,
        aux_logits: bool = True,
        inception_blocks: Optional[List[Callable[..., nn.Module]]] = None,
        init_weights: Optional[bool] = None,
        dropout: float = 0.5
    ) -> None:
        
        super(InceptionV3, self).__init__()
        
        if inception_blocks is None:
            inception_blocks = [
                BasicConv2d,
                InceptionA,
                InceptionB,
                InceptionC,
                InceptionD,
                InceptionE,
                InceptionAux
            ]   
        
        if init_weights is None:
            warnings.warn(
                "The default weight initialization of inception_v3 will be changed in future releases of ""The default weight initialization of inception_v3 will be changed in future releases of "
                "torchvision. If you wish to keep the old behavior (which leads to long initialization times"
                " due to scipy/scipy#11299), please set init_weights=True.",
                FutureWarning,
            )
            init_weights = True
        
        if len(inception_blocks) != 7:
            raise ValueError(f"length of inception block should be 7 instead of {len(inception_blocks)}")
        
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]   
        inception_aux = inception_blocks[6]
        
        self. aux_logtis = aux_logits
        self.Conv2d_1a_3x3 = conv_block(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=2
        )
        self.Conv2d_2a_3x3 = conv_block(
            in_channels=32,
            out_channels=32,
            kernel_size=3
        )
        self.Conv2d_2b_3x3 = conv_block(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=3,
            stride=2
        )
        
        self.Conv2d_3b_1x1 = conv_block(
            in_channels=64,
            out_channels=80,
            kernel_size=1
        )
        self.Conv2d_4a_3x3 = conv_block(
            in_channels=80,
            out_channels=192,
            kernel_size=3
        )
        
        self.maxpool2 = nn.MaxPool2d(
            kernel_size=3,
            stride=2
        )

        self.Mixed_5b = inception_a(
            in_channels=192,
            pool_features=32,
            conv_block=conv_block
        )
        self.Mixed_5c = inception_a(
            in_channels=256,
            pool_features=64,
            conv_block=conv_block
        ) 
        self.Mixed_5d = inception_a(
            in_channels=288,
            pool_features=64,
            conv_block=conv_block
        )
        self.Mixed_6a = inception_b(
            in_channels=288,
            conv_block=conv_block
        )
        self.Mixed_6b = inception_c(
            in_channels=768,
            channels_7x7=128,
            conv_block=conv_block
        )
        self.Mixed_6c = inception_c(
            in_channels=768,
            channels_7x7=160,
            conv_block=conv_block
        )
        self.Mixed_6d = inception_c(
            in_channels = 768,
            channels_7x7=160,
            conv_block=conv_block
        )
        self.Mixed_6e = inception_c(
            in_channels=768,
            channels_7x7=192,
            conv_block=conv_block
        )
        
        self.AuxLogits: Optional[nn.Module] = None
        
        if aux_logits:
            self.AuxLogits = inception_aux(
                in_channels=768,
                num_classes=num_classes,
            )
        
        self.Mixed_7a = inception_d(
            in_channels=768,
            conv_block=conv_block
        )
        self.Mixed_7b = inception_e(
            in_channels=1280,
            conv_block=conv_block
        )
        self.Mixed_7c = inception_e(
            in_channels=2048,
            conv_block=conv_block
        )
        self.avgpool = nn.AdaptiveAvgPool2d(
            output_size=(1, 1)
        )
        self.dropout = nn.Dropout(
            p=dropout,
            inplace=True
        )
        self.fc = nn.Linear(
            in_features=2048,
            out_features=num_classes
        )
        
        
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or \
                    isinstance(m, nn.Linear):
                        stddev = float(m.stddev) if hasattr(m, 'stddev') else 0.1
                        torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=stddev, a=-2,b=2)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def _forward(self, x:Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        #Input: N x 3 x 299 x 299 ->
        x = self.Conv2d_1a_3x3(x) # stride 2
        
        #Output: N x 32 x 149 x 149 ->
        x = self.Conv2d_2a_3x3(x)
        
        #Output: N x 32 x 147 x 147 ->
        x = self.Conv2d_2b_3x3(x) # stride 1
        
        #Output: N x 64 x 147 x 147 ->
        x = self.maxpool1(x) # stride 2
        
        #Output: N x 64 x 73 x 73 ->
        x = self.Conv2d_3b_1x1(x)
        
        #Output: N x 80 x 73 x 73 ->
        x = self.Conv2d_4a_3x3(x) # stride 1
        
        #Output: N x 192 x 71 x 71 ->
        x = self.maxpool2(x) # stride 2
        
        #Output: N x 192 x 35 x 35 ->
        x = self.Mixed_5b(x)
        
        #Output: N x 256 x 35 x 35 ->
        x = self.Mixed_5c(x)
        
        #Output: N x 288 x 35 x 35 ->
        x = self.Mixed_5d(x)
        
        #Output: N x 288 x 35 x 35 ->
        x = self.Mixed_6a(x)
        
        # Output: N x 768 x 17 x 17 ->
        x = self.Mixed_6b(x)
        
        #Output: N x 768 x 17 x 17 ->
        x = self.Mixed_6c(x)
        
        #Output: N x 768 x 17 x 17 ->
        x = self.Mixed_6d(x)
        
        #Output: N x 768 x 17 x 17 ->
        x = self.Mixed_6e(x)
        
        #Output: N x 768 x 17 x 17 ->
        aux: Optional[Tensor] = None
        
        if self.AuxLogits is not None:
            if self.training:
                aux = self.AuxLogits(x)
        
        #Output: N x 768 x 17 x 17 ->
        x = self.Mixed_7a(x)
        
        #Output: N x 1280 x 8 x 8 ->
        x = self.Mixed_7b(x) # stride 2
        
        #Output: N x 2048 x 8 x 8 ->
        x = self.Mixed_7c(x)
        
        #Output: N x 2048 x 8 x 8 ->
        x = self.avgpool(x)
        
        #Output: N x 2048 x 1 x 1 ->
        x = self.dropout(x)
        
        #Output: N x 2048 x 1 x 1 ->
        x = torch.flatten(x, 1)
        
        #Output: N x 2048 ->
        x = self.fc(x)
    
        
        #Output: N x 2 ->
        return x, aux

    # This decorator is ignore and replace with raise exception 
    @torch.jit.unused
    def eager_outputs(
        self,
        x: Tensor,
        aux: Optional[Tensor]
    ) -> InceptionOutputs:
        if self.training and self.aux_logtis:
            return InceptionOutputs(x, aux)
        else:
            return x # type: ignore return value
    
    def forward(self, x: Tensor) -> InceptionOutputs:
        x, aux = self._forward(x)
        aux_defined = self.training and self.aux_logtis
        
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted InceptionV3 always returns InceptionV3 Tuple")
            return InceptionOutputs(x, aux)
        else:
            return self.eager_outputs(x, aux)
    
