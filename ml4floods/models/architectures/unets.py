import torch
import torch.nn as nn

from ml4floods.models.architectures import layer_factory


class UNet(nn.Module):
    def __init__(self, n_channels, n_class):
        super().__init__()

        self.dconv_down1 = layer_factory.double_conv(n_channels, 64)
        self.dconv_down2 = layer_factory.double_conv(64, 128)
        self.dconv_down3 = layer_factory.double_conv(128, 256)
        self.dconv_down4 = layer_factory.double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)

        self.dconv_up3 = layer_factory.double_conv(256 + 512, 256)
        self.dconv_up2 = layer_factory.double_conv(128 + 256, 128)
        self.dconv_up1 = layer_factory.double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out

    
class UNet_dropout(nn.Module):
    def __init__(self, n_channels, n_class):
        super().__init__()

        self.dconv_down1 = layer_factory.double_conv(n_channels, 64)
        self.dconv_down2 = layer_factory.double_conv(64, 128)
        self.dconv_down3 = layer_factory.double_conv(128, 256)
        self.dconv_down4 = layer_factory.double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)

        self.dconv_up3 = layer_factory.double_conv(256 + 512, 256)
        self.dconv_up2 = layer_factory.double_conv(128 + 256, 128)
        self.dconv_up1 = layer_factory.double_conv(128 + 64, 64)
        
        self.dropout = nn.Dropout2d()

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        conv1 = self.dropout(self.dconv_down1(x))
        x = self.maxpool(conv1)

        conv2 = self.dropout(self.dconv_down2(x))
        x = self.maxpool(conv2)

        conv3 = self.dropout(self.dconv_down3(x))
        x = self.maxpool(conv3)

        x = self.dropout(self.dconv_down4(x))

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv3], dim=1)

        x = self.dropout(self.dconv_up3(x))
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv2], dim=1)

        x = self.dropout(self.dconv_up2(x))
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv1], dim=1)

        x = self.dropout(self.dconv_up1(x))

        out = self.conv_last(x)

        return out
