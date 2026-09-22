import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class MSAG(nn.Module):
    """
    Multi-Scale Spatial Attention Gate (MSAG).
    Focuses the model on lesions (exudates, hemorrhages) by computing a spatial attention map.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_map = self.conv(x)
        return x * attention_map

class DRISHYAStudentMTL(nn.Module):
    """
    State-of-the-Art DRISHYA Student Model (V2).
    Backbone: EfficientNetV2-B0 (Edge optimized)
    Decoder: U-Net with MSAG (Spatial Attention)
    """
    def __init__(self, backbone_name='tu-tf_efficientnetv2_b0', pretrained=True, num_classes=5, num_masks=4):
        super().__init__()
        
        encoder_weights = "imagenet" if pretrained else None
        
        self.unet = smp.Unet(
            encoder_name=backbone_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_masks,
            # We use SCSE in the decoder for built-in channel/spatial attention
            decoder_attention_type='scse'
        )
        
        encoder_out_dim = self.unet.encoder.out_channels[-1]
        
        # Classification Head (Standard, Student doesn't strictly need CORAL head if KD handles logits, 
        # but we can use it. We'll use standard since KD distills K logits)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(encoder_out_dim, num_classes)
        )
        
        # Custom MSAG applied before final segmentation output
        # In SMP, decoder_output has channels = decoder_channels[-1] (usually 16)
        decoder_out_channels = 16
        self.msag = MSAG(in_channels=decoder_out_channels)

    def forward(self, x):
        features = self.unet.encoder(x)
        
        final_feature = features[-1]
        pooled = self.global_pool(final_feature).flatten(1)
        logits = self.classifier(pooled)
        
        decoder_output = self.unet.decoder(features)
        
        # Apply MSAG to enhance lesion features before final mask projection
        attended_features = self.msag(decoder_output)
        masks = self.unet.segmentation_head(attended_features)
        
        return logits, masks

if __name__ == "__main__":
    print("Testing DRISHYAStudentMTL V2 initialization...")
    model = DRISHYAStudentMTL(pretrained=False)
    dummy_input = torch.randn(2, 3, 384, 384)
    logits, masks = model(dummy_input)
    print(f"Logits shape: {logits.shape}")
    print(f"Masks shape: {masks.shape}")
    print("Success! V2 Student architecture is ready.")
