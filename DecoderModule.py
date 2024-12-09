import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttModule(nn.Module):
    """ Channel attention module """
    def __init__(self, channels, ratio=16):
        super(ChannelAttModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // ratio, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(channels // ratio, channels, kernel_size=1, bias=False)
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
        att = self.sig(self.conv(torch.cat([avg_out, max_out], dim=1)))
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
    """ Takes in the high res and low res channels to perform one iteration of BAF """
    def __init__(self, high_res, low_res, output, ratio=16, kernel_size=7):
        super(BAFSubModule, self).__init__()
        self.high_res_att = FullAttModel(high_res, ratio, kernel_size)
        self.low_res_att = FullAttModel(low_res, ratio, kernel_size)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(high_res + low_res, output, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x_high, x_low):
        x_high = self.high_res_att(x_high)
        x_low = self.low_res_att(x_low)
        # if x_high.shape[2:] != x_low.shape[2:]:   # check if demensions for high res and low res match
        #     x_high = F.interpolate(x_high, size=x_low.shape[2:], mode='bilinear', align_corners=False)
        x_cat = torch.cat([x_high, x_low], dim=1)
        # print(f"x_high: {x_high.shape}, x_low: {x_low.shape}, x_cat: {x_cat.shape}")
        x = self.relu(self.conv(x_cat))
        return x
    


class DecoderModule(nn.Module):
    """ Full decoder module """
    def __init__(self, channels):
        super(DecoderModule, self).__init__()
        self.baf1 = BAFSubModule(channels[1], channels[0], channels[0])   # 1/8 and 1/4
        self.baf2 = BAFSubModule(channels[2], channels[1], channels[1])   # 1/16 and 1/8
        self.baf3 = BAFSubModule(channels[3], channels[2], channels[2])   # 1/32 and 1/16
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        #self.conv = nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1)
        #self.relu = nn.ReLU()
        
        self.final_layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(24, 24, kernel_size=3, padding=1), # for extra parameters and depth
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(24, 12, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2, x3, x4):
        # print(f"x1: {x1.shape}, x2: {x2.shape}, x3: {x3.shape}, x4: {x4.shape}")
        x_baf3 = self.baf3(x4, x3)  # 1/32 to 1/16]
        x_baf3 = self.upsample(x_baf3)
        print("x_baf3:", x_baf3.shape)
        x_baf2 = self.baf2(x_baf3, x2)  # 1/16 to 1/8
        #x_baf2 = self.upsample(x_baf2)
        print("x_baf2:", x_baf2.shape)
        x_baf1 = self.baf1(x_baf2, x1)  # 1/8 to 1/4
        #x = F.interpolate(x_baf1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        print("x_baf1:", x_baf1.shape)
        x = self.final_layer(x_baf1)
        return x