import torch
import torch.nn as nn


# UNET parts
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(Decoder, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
        self.batch = nn.BatchNorm2d(out_ch)

    def forward(self, x1, x2):
        x1 = self.up_conv(x1)
        x = torch.cat((x2, x1), dim=1)
        x = self.conv(x)
        return x


# UNET
class UNET(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNET, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = Encoder(n_channels, 64)
        self.enc2 = Encoder(64, 128)
        self.enc3 = Encoder(128, 256)
        self.enc4 = Encoder(256, 512)
        self.double_conv1 = DoubleConv(512, 1024)

        self.dec1 = Decoder(1024, 512, 512)
        self.dec2 = Decoder(512, 256, 256)
        self.dec3 = Decoder(256, 128, 128)
        self.dec4 = Decoder(128, 64, 64)
        self.conv = nn.Conv2d(64, n_classes, 1)

    def forward(self, x, out=None):
        # Down
        x1 = self.enc1(x)
        x11 = self.pool(x1)

        x2 = self.enc2(x11)
        x22 = self.pool(x2)

        x3 = self.enc3(x22)
        x33 = self.pool(x3)

        x4 = self.enc4(x33)
        x44 = self.pool(x4)

        x5 = self.double_conv1(x44)

        # Up
        x6 = self.dec1(x5, x4)
        x7 = self.dec2(x6, x3)
        x8 = self.dec3(x7, x2)
        x9 = self.dec4(x8, x1)
        x10 = self.conv(x9)

        return x10
