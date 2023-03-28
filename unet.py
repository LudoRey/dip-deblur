import torch.nn as nn
from torch import cat

def simple_block(in_channels, out_channels, kernel_size=3):
    padding = kernel_size//2
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
    return block

def down_block(in_channels, out_channels, kernel_size=3):
    padding = kernel_size//2
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
    return block

def up_block(in_channels, out_channels, kernel_size=3):
    padding = kernel_size//2
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),  
        nn.Upsample(scale_factor=2, mode='bilinear')
    )
    return block

def skip_block(in_channels, out_channels, kernel_size=1):
    padding = kernel_size//2
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
    return block


class Unet(nn.Module):
    def __init__(self,
                in_channels = 32,
                down_channels = 128,
                up_channels = 128,
                skip_channels = 4,
                out_channels = 3,
                depth = 4):
        super().__init__()

        self.depth = depth
        self.enc_blocks = [] # of length depth+1
        self.dec_blocks = [] # of length depth+1
        self.skip_blocks = [] # of length depth

        self.enc_blocks.append(simple_block(in_channels, down_channels))
        self.dec_blocks.append(simple_block(skip_channels+up_channels, up_channels))

        for l in range(self.depth):
            # Down blocks
            self.enc_blocks.append(down_block(down_channels, down_channels))

            # Up blocks
            if l == self.depth-1:
                self.dec_blocks.append(up_block(down_channels, up_channels))
            else:
                self.dec_blocks.append(up_block(skip_channels+up_channels, up_channels))

            # Skip blocks
            self.skip_blocks.append(skip_block(down_channels, skip_channels))

        self.out_block = nn.Conv2d(up_channels, out_channels, kernel_size=1)
        
        
    def forward(self, x):
        # Encoder + store skips
        skips = []
        for l in range(self.depth):
            x = self.enc_blocks[l](x)
            skips.append(self.skip_blocks[l](x))
        # Bottom block
        x = self.enc_blocks[self.depth](x)
        x = self.dec_blocks[self.depth](x)
        # Decoder
        for l in reversed(range(self.depth)):
            x = self.dec_blocks[l](cat([x, skips[l]], dim=1))
        return self.out_block(x)