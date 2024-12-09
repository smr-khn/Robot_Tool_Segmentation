import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttModule(nn.Module):
    """ Channel attention module """
    def __init__(self, channels, ratio=16):
        super(ChannelAttModule, self).__init__()
        reduced_channels = max(1, channels // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return x * self.sig(avg_out + max_out)   # in paper computational graph, it shows refeeding of x, but in their github code, does not
    


class SpatialAttModule(nn.Module):
    """ Spatial attention module """
    def __init__(self, kernel_size=7):
        super(SpatialAttModule, self).__init__()
        assert kernel_size in (3, 7)
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        att = torch.cat([avg_out, max_out], dim=1)
        att = self.sig(self.conv(att))
        return x * att   # in paper computational graph, it shows refeeding of x, but in their github code, does not
    


class FullAttModel(nn.Module):
    """ Attention model that combines the channel and spatial attention sub models """
    def __init__(self, channels, ratio=16, kernel_size=7):
        super(FullAttModel, self).__init__()
        self.channel_att = ChannelAttModule(channels, ratio)
        self.spatial_att = SpatialAttModule(kernel_size)

    def forward(self, x):
        # print(f"Inp to FullAtt: {x.shape}")
        x = self.channel_att(x)
        # print(f"Aft ChannelAtt: {x.shape}")
        x = self.spatial_att(x)
        # print(f"Aft SpatialAtt: {x.shape}")
        return x
    


class BAFSubModule(nn.Module):
    def __init__(self, channels_high, channels_low):
        super(BAFSubModule, self).__init__()
        self.channel_att_high = ChannelAttModule(channels_high)
        self.channel_att_low = ChannelAttModule(channels_low)
        self.spatial_att_high = SpatialAttModule(kernel_size=7)
        self.spatial_att_low = SpatialAttModule(kernel_size=7)
        # Downsample high-res to match low-res
        self.conv_high = nn.Conv2d(channels_high, channels_low, kernel_size=3, padding=1, stride=1)
        self.conv_low = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, stride=1)

    def forward(self, x_high, x_low):
        # attention for high res low res
        x_high_att = self.channel_att_high(x_high)
        x_high_att = self.spatial_att_high(x_high_att)
        x_low_att = self.channel_att_low(x_low)
        x_low_att = self.spatial_att_low(x_low_att)
        x_high_conv = F.interpolate(self.conv_high(x_high_att), size=x_low_att.shape[2:], mode='bilinear', align_corners=False)
        # Fuse high and low-res
        x_fused = x_high_conv + self.conv_low(x_low_att)
        # print(f"BAFSubModule output: {x_fused.shape}")
        return x_fused
    


class DecoderModule(nn.Module):
    """ Full decoder module """
    def __init__(self, channels):
        super(DecoderModule, self).__init__()
        self.baf1 = BAFSubModule(channels[1], channels[0])   # 1/8 and 1/4
        self.baf2 = BAFSubModule(channels[2], channels[1])   # 1/16 and 1/8
        self.baf3 = BAFSubModule(channels[3], channels[2])   # 1/32 and 1/16
        self.final_conv = nn.Conv2d(channels[0], 12, kernel_size=1)

    def forward(self, y1, y2, y3, y4):
        # First BAF
        out3 = self.baf3(x_high=y4, x_low=y3)  
        # print(f"out3 shape: {out3.shape}")   # [1, 24, 80, 56]
        out3_upsample = F.interpolate(out3, size=(80, 56), mode='bilinear', align_corners=False)
        # Second BAF
        out2 = self.baf2(x_high=out3_upsample, x_low=y2)  
        # print(f"out2 shape: {out2.shape}")   # [1, 32, 40, 32]
        out2_upsample = F.interpolate(out2, size=(40, 32), mode='bilinear', align_corners=False)
        # Third BAF
        out1 = self.baf1(x_high=out2_upsample, x_low=y1)  
        # print(f"out1 shape: {out1.shape}")   # [1, 64, 20, 16]
        out1_upsample = F.interpolate(out1, size=(20, 16), mode='bilinear', align_corners=False)
        final_out = F.interpolate(out1_upsample, size=(320, 256), mode='bilinear', align_corners=False)
        return self.final_conv(final_out)