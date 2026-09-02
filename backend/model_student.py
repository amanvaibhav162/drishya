"""
DRISHYA AI — PP-LCNet Multi-Task Student Model Architecture
Distilled from a 15-model Teacher Ensemble (EfficientNet-B4 + ConvNeXt-Tiny + RegNetY-040).
Outputs:
  - 5-class ICDR DR grading logits
  - 4-channel lesion segmentation masks (MA, EX, HE, SE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class MiniUNetDecoder(nn.Module):
    """
    A ridiculously lightweight U-Net decoder designed specifically to attach to PP-LCNet's
    feature maps without bloating the parameter count or slowing down inference.
    """
    def __init__(self, encoder_channels, out_channels=4):
        super().__init__()
        # encoder_channels for lcnet_100: [32, 64, 128, 256, 512]

        self.up4 = nn.ConvTranspose2d(encoder_channels[4], encoder_channels[3], kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(encoder_channels[3] * 2, encoder_channels[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(encoder_channels[3]),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.ConvTranspose2d(encoder_channels[3], encoder_channels[2], kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(encoder_channels[2] * 2, encoder_channels[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(encoder_channels[2]),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(encoder_channels[2], encoder_channels[1], kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(encoder_channels[1] * 2, encoder_channels[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(encoder_channels[1]),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.ConvTranspose2d(encoder_channels[1], encoder_channels[0], kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(encoder_channels[0] * 2, encoder_channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(encoder_channels[0]),
            nn.ReLU(inplace=True)
        )

        # Final upsample from Stride 2 to Stride 1 (384x384)
        self.final_up = nn.ConvTranspose2d(encoder_channels[0], encoder_channels[0], kernel_size=2, stride=2)

        # Output 4 masks (MA, EX, HE, SE/Vessels)
        self.final_conv = nn.Conv2d(encoder_channels[0], out_channels, kernel_size=1)

    def forward(self, features):
        c1, c2, c3, c4, c5 = features

        # Decode 5 -> 4
        d4 = self.up4(c5)
        d4 = torch.cat([d4, c4], dim=1)
        d4 = self.conv4(d4)

        # Decode 4 -> 3
        d3 = self.up3(d4)
        d3 = torch.cat([d3, c3], dim=1)
        d3 = self.conv3(d3)

        # Decode 3 -> 2
        d2 = self.up2(d3)
        d2 = torch.cat([d2, c2], dim=1)
        d2 = self.conv2(d2)

        # Decode 2 -> 1
        d1 = self.up1(d2)
        d1 = torch.cat([d1, c1], dim=1)
        d1 = self.conv1(d1)

        # Decode to full resolution
        out = self.final_up(d1)
        out = self.final_conv(out)

        return out


class DRISHYAStudentMTL(nn.Module):
    """
    The Ultra-Lightweight Student Model (PP-LCNet Backbone)
    Designed for Knowledge Distillation and Edge Inference (< 15MB weights).
    """
    def __init__(self, num_classes=5, num_masks=4):
        super().__init__()

        # We use features_only=True to extract intermediate layers for the decoder
        self.encoder = timm.create_model('lcnet_100', pretrained=False, features_only=True)

        # PP-LCNet specific feature channels: [32, 64, 128, 256, 512]
        self.encoder_channels = [32, 64, 128, 256, 512]

        # Classification Head (Global Average Pooling -> Linear)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.encoder_channels[-1], num_classes)

        # Segmentation Head (Custom Lightweight U-Net Decoder)
        self.decoder = MiniUNetDecoder(self.encoder_channels, out_channels=num_masks)

    def forward(self, x):
        # Extract features from the PP-LCNet backbone
        features = self.encoder(x)

        # Classification (using the deepest feature map 'c5')
        c5 = features[-1]
        pooled = self.global_pool(c5).flatten(1)
        logits = self.classifier(pooled)

        # Segmentation (using all feature maps)
        masks = self.decoder(features)

        return logits, masks
