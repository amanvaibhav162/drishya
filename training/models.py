"""
DRISHYA SOTA Training: Multi-Task Model Architectures
Supports:
  - Teacher 1: ConvNeXtV2-Base (tu-convnextv2_base)
  - Teacher 2: Swin-Base-384 (tu-swin_base_patch4_window12_384)
  - Teacher 3: EfficientNet-B5 (tu-efficientnet_b5)
  - Student:   EfficientNetV2-S (tu-tf_efficientnetv2_s.in21k_ft_in1k)

Each architecture outputs:
  1. Ordinal Logits: Shape (B, num_classes - 1) = (B, 4) for CORAL rank-consistent grading
  2. Dense Lesion Masks: Shape (B, num_masks, H, W) = (B, 4, H, W) for MA, EX, HE, SE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Tuple, List, Optional
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base import SegmentationHead

# Safe GlobalResponseNorm monkey-patch to prevent FP16 subnormal underflows & divide-by-zero
try:
    from timm.layers.grn import GlobalResponseNorm
    def _safe_grn_forward(self, x):
        orig_dtype = x.dtype
        x_f32 = x.float()
        x_g = x_f32.norm(p=2, dim=self.spatial_dim, keepdim=True)
        x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + 1e-5)
        res = x_f32 + torch.addcmul(
            self.bias.float().view(self.wb_shape),
            self.weight.float().view(self.wb_shape),
            x_f32 * x_n
        )
        return res.to(orig_dtype)
    GlobalResponseNorm.forward = _safe_grn_forward
except Exception:
    pass



class MSAG(nn.Module):
    """
    Multi-Scale Attention Gate for lesion localization and feature refinement.
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


class DRISHYAMultiTaskModel(nn.Module):
    """
    Generalized Multi-Task Model:
    Backbone (timm features_only) + scSE UNet Decoder + CORAL Ordinal Head.
    """
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool = True,
        num_classes: int = 5,
        num_masks: int = 4,
        img_size: int = 512,
        decoder_channels: Optional[List[int]] = None,
        use_coral: bool = True,
        dropout: float = 0.2,
        gradient_checkpointing: bool = False
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.num_masks = num_masks
        self.use_coral = use_coral
        self.img_size = img_size

        # Output units: 4 for CORAL ordinal regression, 5 for standard multi-class CE
        self.num_logits = (num_classes - 1) if use_coral else num_classes

        # 1. Create backbone encoder
        create_kwargs = {"features_only": True, "pretrained": pretrained}
        try:
            self.encoder = timm.create_model(backbone_name, **create_kwargs)
        except Exception:
            # Fallback if specific pretrained weights name is needed
            fallback_name = backbone_name.replace("tu-", "")
            self.encoder = timm.create_model(fallback_name, **create_kwargs)

        if gradient_checkpointing and hasattr(self.encoder, 'set_grad_checkpointing'):
            self.encoder.set_grad_checkpointing(enable=True)

        # 2. Dynamically determine feature channel shapes using dummy forward pass
        probe_size = 384 if 'swin' in backbone_name.lower() else self.img_size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, probe_size, probe_size)
            raw_feats = self.encoder(dummy)
            feats = [f.permute(0, 3, 1, 2) if (f.ndim == 4 and f.shape[1] != f.shape[-1] and f.shape[-1] in [96, 128, 192, 256, 384, 512, 768, 1024, 1536]) else f for f in raw_feats]
            encoder_channels = [3] + [f.shape[1] for f in feats]
            bottleneck_channels = feats[-1].shape[1]

        # 3. Construct scSE UNet Decoder
        num_stages = len(feats)
        if decoder_channels is None:
            # Scale decoder channels based on stages
            default_channels = [256, 128, 64, 32, 16]
            decoder_channels = default_channels[:num_stages]

        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=num_stages,
            use_norm='batchnorm',
            attention_type='scse',
            add_center_block=False
        )

        # 4. Segmentation Head
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=num_masks,
            kernel_size=3
        )

        # 5. Classification / Ordinal Head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(bottleneck_channels, self.num_logits)
        )

        self.msag = MSAG(in_channels=decoder_channels[-1], mid_channels=8)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with multi-resolution Swin/ConvNeXt adaptability.
        Args:
            x: Input tensor [B, 3, H, W]
        Returns:
            logits: [B, num_logits] (CORAL ordinal binary logits or class logits)
            masks:  [B, num_masks, H, W] raw segmentation logits at native (H, W)
        """
        orig_h, orig_w = x.shape[2], x.shape[3]
        if 'swin' in self.backbone_name.lower() and (orig_h != 384 or orig_w != 384):
            x_enc = F.interpolate(x, size=(384, 384), mode='bilinear', align_corners=False)
        else:
            x_enc = x

        # Multi-scale feature extraction (handling channels-last Swin formats)
        raw_feats = self.encoder(x_enc)
        features = [f.permute(0, 3, 1, 2) if (f.ndim == 4 and f.shape[1] != f.shape[-1] and f.shape[-1] in [96, 128, 192, 256, 384, 512, 768, 1024, 1536]) else f for f in raw_feats]

        # UNet decoder reconstruction executed in FP32 to prevent BatchNorm running stats overflow
        with torch.amp.autocast('cuda', enabled=False):
            features_f32 = [f.float() for f in features]
            dec_inputs = [x_enc.float()] + features_f32
            decoder_out = self.decoder(dec_inputs)

            # Apply MSAG attention weighting
            att_weights = self.msag(decoder_out)
            refined_decoder_out = decoder_out * att_weights
            masks = self.segmentation_head(refined_decoder_out)

            # Restore native resolution (512x512) if Swin was downsampled to 384
            if masks.shape[2:] != (orig_h, orig_w):
                masks = F.interpolate(masks, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

            # Global pooling & ordinal/classification head
            bottleneck = features_f32[-1]
            pooled = F.adaptive_avg_pool2d(bottleneck, 1).flatten(1)
            logits = self.classifier(pooled)

        return logits, masks

    def predict_class_probabilities(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs forward pass and converts outputs to normalized 5-class probabilities
        and sigmoid lesion probability masks.
        """
        logits, mask_logits = self.forward(x)
        if self.use_coral:
            probs = torch.sigmoid(logits)  # [B, 4]
            p0 = 1.0 - probs[:, 0:1]
            p1 = torch.clamp(probs[:, 0:1] - probs[:, 1:2], min=0.0)
            p2 = torch.clamp(probs[:, 1:2] - probs[:, 2:3], min=0.0)
            p3 = torch.clamp(probs[:, 2:3] - probs[:, 3:4], min=0.0)
            p4 = probs[:, 3:4]
            class_probs = torch.cat([p0, p1, p2, p3, p4], dim=-1)
            class_probs = class_probs / (class_probs.sum(dim=-1, keepdim=True) + 1e-7)
        else:
            class_probs = F.softmax(logits, dim=-1)

        mask_probs = torch.sigmoid(mask_logits)
        return class_probs, mask_probs


def build_teacher_model(teacher_id: int, pretrained: bool = True, gradient_checkpointing: bool = True) -> DRISHYAMultiTaskModel:
    """
    Builds one of the 3 orthogonal teacher backbones calibrated for 24 GB VRAM:
      Teacher 1: ConvNeXtV2-Base
      Teacher 2: Swin-Base-384
      Teacher 3: EfficientNet-B5
    """
    if teacher_id == 1:
        arch = "convnextv2_base"
        gradient_checkpointing = True
    elif teacher_id == 2:
        arch = "swin_base_patch4_window12_384"
        gradient_checkpointing = True
    elif teacher_id == 3:
        arch = "efficientnet_b5"
        gradient_checkpointing = True
    else:
        raise ValueError(f"Unknown teacher_id {teacher_id}. Must be 1, 2, or 3.")

    return DRISHYAMultiTaskModel(
        backbone_name=arch,
        pretrained=pretrained,
        num_classes=5,
        num_masks=4,
        img_size=512,
        use_coral=True,
        gradient_checkpointing=gradient_checkpointing
    )


def build_student_model(pretrained: bool = True) -> DRISHYAMultiTaskModel:
    """
    Builds the production EfficientNetV2-S student model.
    Target: 22.09M parameters, 42.1 MB FP16 ONNX, <12 ms edge latency.
    """
    return DRISHYAMultiTaskModel(
        backbone_name="tf_efficientnetv2_s.in21k_ft_in1k",
        pretrained=pretrained,
        num_classes=5,
        num_masks=4,
        img_size=512,
        use_coral=True,
        gradient_checkpointing=False
    )
