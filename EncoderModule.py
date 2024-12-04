import torch
import torch.nn as nn
from torchvision import models

class MobileNetV2Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetV2Encoder, self).__init__()

        # Load the MobileNetV2 model
        mobilenet_v2 = models.mobilenet_v2(pretrained=pretrained)
        
        # Extract the feature layers
        self.features = mobilenet_v2.features
        
        self.map1 = nn.Sequential(*self.features[:4])  #1/4 res
        self.map2 = nn.Sequential(*self.features[4:7]) #1/8 res
        self.map3 = nn.Sequential(*self.features[7:11]) #1/16 res
        self.map4 = nn.Sequential(*self.features[11:])  #1/32 res

    def forward(self, x):
        x1 = self.map1(x) 
        x2 = self.map2(x1)
        x3 = self.map3(x2)
        x4 = self.map4(x3)
        
        return x1, x2, x3, x4