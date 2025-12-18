import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # 编码器部分
        self.encoder1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 中间部分
        self.middle = DoubleConv(512, 1024)

        # 解码器部分
        self.decoder4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder_conv4 = DoubleConv(1024, 512)
        self.decoder3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder_conv3 = DoubleConv(512, 256)
        self.decoder2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_conv2 = DoubleConv(256, 128)
        self.decoder1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_conv1 = DoubleConv(128, 64)

        # 最后的卷积层，将通道数调整为输出要求的通道数
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 编码器部分
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # 中间部分
        middle = self.middle(self.pool4(enc4))

        # 解码器部分
        dec4 = self.decoder4(middle)
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.decoder_conv4(dec4)
        dec3 = self.decoder3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.decoder_conv3(dec3)
        dec2 = self.decoder2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.decoder_conv2(dec2)
        dec1 = self.decoder1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.decoder_conv1(dec1)

        # 最后的卷积层
        final_output = self.final_conv(dec1) + x[:, :3, :, :]
        final_output = torch.clamp(final_output, 0, 1)

        return final_output
################################################