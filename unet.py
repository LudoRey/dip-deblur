import torch.nn as nn
from torch import cat

class SimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size//2
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size//2
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size//2
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.up(x)
        return x

class SkipBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        padding = kernel_size//2
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class OutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        padding = kernel_size//2
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.sig(x)
        return x

class Unet(nn.Module):
    def __init__(self,
                in_channels = 32,
                feature_channels = 128,
                feature_ksize = 3,
                skip_channels = 4,
                skip_ksize = 1,
                out_channels = 3,
                depth = 4):
        super().__init__()

        self.depth = depth
        self.enc_blocks = nn.ModuleList([]) # of length depth+1
        self.dec_blocks = nn.ModuleList([]) # of length depth+1
        self.skip_blocks = nn.ModuleList([]) # of length depth

        self.enc_blocks.append(SimpleBlock(in_channels, feature_channels, kernel_size=feature_ksize))
        self.dec_blocks.append(SimpleBlock(skip_channels+feature_channels, feature_channels, kernel_size=feature_ksize))

        for l in range(self.depth):
            # Down blocks
            self.enc_blocks.append(DownBlock(feature_channels, feature_channels, kernel_size=feature_ksize))

            # Up blocks
            if l == self.depth-1:
                self.dec_blocks.append(UpBlock(feature_channels, feature_channels, kernel_size=feature_ksize))
            else:
                self.dec_blocks.append(UpBlock(skip_channels+feature_channels, feature_channels, kernel_size=feature_ksize))

            # Skip blocks
            if skip_channels != 0:
                self.skip_blocks.append(SkipBlock(feature_channels, skip_channels, kernel_size=skip_ksize))

        self.out_block = OutBlock(feature_channels, out_channels, kernel_size=1)
        
        
    def forward(self, x):
        # Encoder + store skips
        skips = []
        for l in range(self.depth):
            x = self.enc_blocks[l](x)
            if len(self.skip_blocks) != 0:
                skips.append(self.skip_blocks[l](x))
        # Bottom block
        x = self.enc_blocks[self.depth](x)
        x = self.dec_blocks[self.depth](x)
        # Decoder
        for l in reversed(range(self.depth)):
            if len(self.skip_blocks) != 0:
                x = self.dec_blocks[l](cat([x, skips[l]], dim=1))
            else:
                x = self.dec_blocks[l](x)
        return self.out_block(x)