"""
DRISHYA AI — Multi-Task Learning Student Model Architecture
Features:
  - EfficientNetV2-B0 backbone (timm) with Fused-MBConv & Inverted Residual stages
  - UNet Decoder with Spatial and Channel Squeeze & Excitation (scSE) Attention
  - Multi-Scale Attention Gate (MSAG)
  - 5-class ICDR Diabetic Retinopathy classification head
  - 4-channel pixel-level lesion segmentation head (MA, EX, HE, SE)
"""

import os
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base import SegmentationHead


class MSAG(nn.Module):
    """
    Multi-Scale Attention Gate (MSAG) for lesion localization and feature refinement.
    """
    def __init__(self, in_channels: int = 16, mid_channels: int = 8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class EfficientNetEncoderWrapper(nn.Module):
    """
    Wrapper around timm EfficientNetV2 encoder matching checkpoint key prefix 'unet.encoder.model...'
    """
    def __init__(self, model_name: str = 'tf_efficientnetv2_b0'):
        super().__init__()
        self.model = timm.create_model(model_name, features_only=True, pretrained=False)

    def forward(self, x: torch.Tensor):
        return self.model(x)


class EfficientNetUnetWrapper(nn.Module):
    """
    UNet wrapper matching checkpoint keys:
    unet.encoder.model..., unet.decoder..., unet.segmentation_head...
    """
    def __init__(self, num_masks: int = 4):
        super().__init__()
        self.encoder = EfficientNetEncoderWrapper('tf_efficientnetv2_b0')
        self.decoder = UnetDecoder(
            encoder_channels=[3, 16, 32, 48, 112, 192],
            decoder_channels=[256, 128, 64, 32, 16],
            n_blocks=5,
            use_norm='batchnorm',
            attention_type='scse',
            add_center_block=False
        )
        self.segmentation_head = SegmentationHead(
            in_channels=16, out_channels=num_masks, kernel_size=3
        )


class DRISHYAEfficientNetV2MTL(nn.Module):
    """
    Production Multi-Task Student Model (EfficientNetV2-B0 + scSE UNet Decoder + MSAG).
    Trained for joint 5-class ICDR grading and 4-channel lesion segmentation.
    Total parameters: ~7.67M.
    """
    arch_name = "EfficientNetV2-B0 Student MTL (7.67M params)"

    def __init__(self, num_classes: int = 5, num_masks: int = 4):
        super().__init__()
        self.unet = EfficientNetUnetWrapper(num_masks=num_masks)
        self.msag = MSAG(in_channels=16, mid_channels=8)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(192, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Feature extraction through EfficientNetV2-B0 encoder
        features = self.unet.encoder.model(x)  # [c1, c2, c3, c4, c5]

        # 2. Reconstruct high-resolution lesion masks via scSE UNet Decoder
        dec_features = [x] + features
        decoder_out = self.unet.decoder(dec_features)  # (B, 16, H, W)
        masks = self.unet.segmentation_head(decoder_out)  # (B, 4, H, W)

        # 3. Global pooling & classification
        pooled = F.adaptive_avg_pool2d(features[-1], 1).flatten(1)  # (B, 192)
        logits = self.classifier(pooled)  # (B, 5)

        return logits, masks

    def get_target_layer(self) -> nn.Module:
        """Returns the final convolutional block of the encoder for Grad-CAM++."""
        encoder_model = getattr(self.unet.encoder, 'model', self.unet.encoder)
        blocks = getattr(encoder_model, 'blocks', None)
        if blocks is not None:
            return blocks[-1]
        raise AttributeError("Encoder model does not have 'blocks' attribute.")



# Default student class is EfficientNetV2 MTL
DRISHYAStudentMTL = DRISHYAEfficientNetV2MTL


def load_student_model(model_path: str, device: torch.device, num_classes: int = 5, num_masks: int = 4) -> DRISHYAEfficientNetV2MTL:
    """
    Loads the trained DRISHYA Multi-Task Student Model (EfficientNetV2-B0).
    """
    model = DRISHYAEfficientNetV2MTL(num_classes=num_classes, num_masks=num_masks).to(device)

    if not os.path.exists(model_path):
        print(f"[DRISHYA] ✗ Checkpoint {model_path} not found. Instantiating uninitialized EfficientNetV2 MTL.")
        return model

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[DRISHYA] ✓ Loaded trained student model from {model_path} ({model.arch_name})")
    return model

