"""
Implementation of BSRGAN architecture for image super-resolution.

This module contains the core components of the BSRGAN model, including:
- Residual Dense Block (RDB)
- Residual in Residual Dense Block (RRDB) 
- The full RRDBNet architecture

The implementation is based on the paper:
"BSRGAN: Designing a Practical Degradation Model for Deep Blind Image Super-Resolution"
"""

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def initialize_weights(net_l, scale=1):
    """Initialize network weights using Kaiming initialization.
    
    Args:
        net_l (nn.Module or list): Network module(s) to initialize
        scale (float): Scaling factor for weights, used for residual blocks
        
    The function initializes:
    - Conv2d layers with Kaiming normal initialization
    - Linear layers with Kaiming normal initialization  
    - BatchNorm2d layers with constant 1 for weights and 0 for bias
    """
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    """Create a sequential layer composed of multiple blocks.
    
    Args:
        block: Block module to use
        n_layers (int): Number of blocks to stack
        
    Returns:
        nn.Sequential: Sequential layer containing the stacked blocks
    """
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    """Residual Dense Block with 5 convolutions.
    
    A dense block where each layer's input is the concatenation of all previous layers' outputs.
    Includes a residual connection from input to output.
    
    Args:
        nf (int): Number of filters (channels)
        gc (int): Growth channels, i.e. intermediate channels
        bias (bool): Whether to include bias in convolutions
    """
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.
    
    Combines multiple Residual Dense Blocks in a residual manner.
    
    Args:
        nf (int): Number of filters
        gc (int): Growth channels for the ResidualDenseBlock_5C
    """
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """RRDBNet architecture for image super-resolution.
    
    A deep network using Residual in Residual Dense Blocks (RRDB) as its building blocks.
    Supports x2 and x4 upscaling factors.
    
    Args:
        in_nc (int): Number of input channels (default: 3 for RGB)
        out_nc (int): Number of output channels (default: 3 for RGB)
        nf (int): Number of filters
        nb (int): Number of RRDB blocks
        gc (int): Growth channels for RRDB
        sf (int): Scaling factor, either 2 or 4 (default: 4)
    """
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.sf = sf
        print([in_nc, out_nc, nf, nb, gc, sf])

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if self.sf==4:
            self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        if self.sf==4:
            fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out
