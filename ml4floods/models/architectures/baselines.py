import torch.nn as nn

from ml4floods.models.architectures import layer_factory


class SimpleLinear(nn.Module):
    """
    Linear model
    """
    def __init__(self, n_channels, n_class):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, n_class, 1)
        )

    def forward(self, x):
        
        res = self.conv(x)
        
        return res


class SimpleCNN(nn.Module):
    """
    5-layer fully conv CNN
    """
    def __init__(self, n_channels, n_class):
        super().__init__()

        self.conv = nn.Sequential(
            layer_factory.double_conv(n_channels, 64),
            layer_factory.double_conv(64, 128),
            nn.Conv2d(128, n_class, 1)
        )
        
    def forward(self, x):
        
        res = self.conv(x)
        
        return res
