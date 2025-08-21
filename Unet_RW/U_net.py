
import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1):
        super().__init__()
        p = k//2
        self.seq = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, p, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.seq(x)

class UNetBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv1 = ConvBNReLU(c_in, c_out)
        self.conv2 = ConvBNReLU(c_out, c_out)
    def forward(self, x): return self.conv2(self.conv1(x))

class UNet(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.enc1 = UNetBlock(3, ch)
        self.enc2 = UNetBlock(ch, ch*2)
        self.enc3 = UNetBlock(ch*2, ch*4)
        self.enc4 = UNetBlock(ch*4, ch*8)
        self.pool = nn.MaxPool2d(2,2)
        self.up4 = nn.ConvTranspose2d(ch*8, ch*4, 2, 2)
        self.dec4 = UNetBlock(ch*8, ch*4)
        self.up3 = nn.ConvTranspose2d(ch*4, ch*2, 2, 2)
        self.dec3 = UNetBlock(ch*4, ch*2)
        self.up2 = nn.ConvTranspose2d(ch*2, ch, 2, 2)
        self.dec2 = UNetBlock(ch*2, ch)
        self.out = nn.Conv2d(ch, 1, 1)
    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool(e1)
        e2 = self.enc2(p1); p2 = self.pool(e2)
        e3 = self.enc3(p2); p3 = self.pool(e3)
        e4 = self.enc4(p3)
        d4 = self.dec4(torch.cat([self.up4(e4), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        return self.out(d2)
