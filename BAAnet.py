import torch
import torch.nn as nn
import torch.nn.functional as F

import BBAModule
import EncoderModule

class BAANet_BAA_Only(nn.Module):
    '''
    BAAnet with BAF module removed and an adapted output sequence.
    '''
    def __init__(self, pretrained=True):
        super(BAANet_BAA_Only, self).__init__()
        
        self.encoder = EncoderModule.MobileNetV2Encoder(pretrained=pretrained)
        self.BBA = BBAModule.BBAModule([24,32,64,160])
        
        self.convUp1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        
        self.convUp2 = nn.Conv2d(32, 24, kernel_size=3, padding=1)
        
        self.final_layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(24, 24, kernel_size=3, padding=1), # for extra parameters and depth
            nn.Softmax(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(24, 12, kernel_size=3, padding=1),
            nn.Softmax()
        )
        
    def forward(self, x):
        
        x1,x2,x3,x4 = self.encoder(x)
        x4,x3,x2,x1 = self.BBA(x1,x2,x3,x4)
        
        # from paper, modified output layer
        out = x1 + x2
        out = F.interpolate(out, size=x3.shape[2:], mode='bilinear', align_corners=False)
        out = self.convUp1(out)
        out = out + x3
        out = F.interpolate(out, size=x4.shape[2:], mode='bilinear', align_corners=False)
        out = self.convUp2(out)
        out = self.final_layer(out + x4)
        
        return out
    
class BAANet(nn.Module):
    def __init__(self, pretrained=True):
        super(BAANet, self).__init__()
        
        self.encoder = EncoderModule.MobileNetV2Encoder(pretrained=pretrained)
        self.BBA = BBAModule.BBAModule([24,32,64,160])
        
        self.final_layer = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=3, padding=1), # for extra parameters and depth
            nn.ReLU(),
            nn.Conv2d(24, 12, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
    def forward(self, x):
        
        x1,x2,x3,x4 = self.encoder(x)
        x3_low, x2_low, x1_low, x1_high = self.BBA(x1,x2,x3,x4)
        
        return x3_low, x2_low, x1_low, x1_high