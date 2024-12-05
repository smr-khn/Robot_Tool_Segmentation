import torch
import torch.nn as nn
import torch.nn.functional as F

class BBAModule(nn.Module):
    def __init__(self, channels):
        super(BBAModule, self).__init__()

        # Convs for upsampling paths
        self.conv4_up_2 = nn.Conv2d(channels[3], channels[2], kernel_size=3, padding=1)
        self.conv4_up_2_out = nn.Conv2d(channels[3], channels[2], kernel_size=3, padding=1)
        self.conv4_up_4 = nn.Conv2d(channels[3], channels[1], kernel_size=3, padding=1)
        self.conv4_up_8 = nn.Conv2d(channels[3], channels[0], kernel_size=3, padding=1)
 
        self.conv3_up_2 = nn.Conv2d(channels[2], channels[1], kernel_size=3, padding=1)
        self.conv3_up_4 = nn.Conv2d(channels[2], channels[0], kernel_size=3, padding=1)

        self.conv2_up_2 = nn.Conv2d(channels[1], channels[0], kernel_size=3, padding=1)

    def forward(self, x1, x2, x3, x4):

        # Upsample x4 to match x3, x2, and x1 size
        x4_up_2 = F.interpolate(x4, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x4_up_2_conv = self.conv4_up_2(x4_up_2)
        x4_up_2_conv_out = self.conv4_up_2_out(x4_up_2)

        x4_up_4 = F.interpolate(x4, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x4_up_4_conv = self.conv4_up_4(x4_up_4)

        x4_up_8 = F.interpolate(x4, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x4_up_8_conv = self.conv4_up_8(x4_up_8)

        # Upsample x3 to match x2 and x1 szie
        x3_up_2 = F.interpolate(x3, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x3_up_2_conv = self.conv3_up_2(x3_up_2)

        x3_up_4 = F.interpolate(x3, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x3_up_4_conv = self.conv3_up_4(x3_up_4)

        # Upsample x2 to match x1 size
        x2_up_2 = F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x2_up_2_conv = self.conv2_up_2(x2_up_2)

        # Feature combination
        y1 = x1 * (x2_up_2_conv + x3_up_4_conv + x4_up_8_conv)
        y2 = x2 * (x3_up_2_conv + x4_up_4_conv)
        y3 = x3 * x4_up_2_conv
        y4 = x4_up_2_conv_out

        return y1, y2, y3, y4
