import torch.nn as nn
from torch import cat

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

class DIP(nn.Module):
    #TODO not up to date
    def __init__(self,
                 in_channels = 32,
                 down_channels = [128, 128, 128, 128, 128],
                 up_channels = [128, 128, 128, 128, 128],
                 skip_channels = [4, 4, 4, 4, 4],
                 out_channels = 3):
        super().__init__()

        assert len(down_channels) == len(up_channels) == len(skip_channels)
        self.layers = len(down_channels)

        enc_channels = [in_channels] + down_channels
        dec_channels = up_channels + [down_channels[-1]]

        self.down_blocks = []
        self.up_blocks = []
        self.skip_blocks = []

        for l in range(self.layers):
            # Skip blocks
            if skip_channels[l] == 0:
                self.skip_blocks.append(None)
            else:
                self.skip_blocks.append(skip_block(enc_channels[l], skip_channels[l]))
                
            # Down and up blocks
            self.down_blocks.append(down_block(enc_channels[l], enc_channels[l+1]))
            if l != self.layers-1:
                self.up_blocks.append(up_block(skip_channels[l+1]+dec_channels[l+1], dec_channels[l]))
            else:
                # Deepest : no skip connection
                self.up_blocks.append(up_block(dec_channels[l+1], dec_channels[l]))

        self.out_block = nn.Sequential(
            nn.Conv2d(dec_channels[0]+skip_channels[0], out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        skips = []
        for l in range(self.layers):
            if self.skip_blocks[l] == None:
                skips.append(None)
            else:
                skips.append(self.skip_blocks[l](x))
                
            x = self.down_blocks[l](x)
        
        # Decoder
        for l in reversed(range(self.layers)):
            if l == self.layers-1 or skips[l+1] == None:
                x = self.up_blocks[l](x)
            else:
                x = self.up_blocks[l](cat([x, skips[l+1]], dim=1))

        # Output
        if skips[0] == None:
            output = self.out_block(x)
        else:
            output = self.out_block(cat([x, skips[0]], dim=1))

        return output