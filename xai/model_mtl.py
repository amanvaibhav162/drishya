import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class DRISHYAMTLModel(nn.Module):
    """
    Multi-Task Learning (MTL) architecture for Diabetic Retinopathy.
    Simultaneously performs ICDR classification (0-4) and Lesion Segmentation.
    """
    def __init__(self, backbone_name='tu-tf_efficientnet_b4.ns_jft_in1k', pretrained=True, num_classes=5, num_masks=4):
        super().__init__()
        
        # We use segmentation_models_pytorch to build a robust U-Net.
        # The 'tu-' prefix allows SMP to use any timm backbone.
        encoder_weights = "imagenet" if pretrained else None
        
        self.unet = smp.Unet(
            encoder_name=backbone_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_masks,
        )
        
        # Retrieve the channel dimension of the deepest feature map
        encoder_out_dim = self.unet.encoder.out_channels[-1]
        
        # Attach the Classification Head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(encoder_out_dim, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, 3, H, W)
        Returns:
            logits: Tensor of shape (B, num_classes) for DR Grading
            masks: Tensor of shape (B, num_masks, H, W) for Lesion Segmentation
        """
        # 1. Extract feature maps at multiple scales (for skip connections)
        features = self.unet.encoder(x)
        
        # 2. Classification Path (using the deepest feature map)
        final_feature = features[-1]
        pooled = self.global_pool(final_feature).flatten(1)
        logits = self.classifier(pooled)
        
        # 3. Segmentation Path (Decoder processes features + skip connections)
        decoder_output = self.unet.decoder(features)
        masks = self.unet.segmentation_head(decoder_output)
        
        return logits, masks

if __name__ == "__main__":
    # Quick sanity check for the model initialization and tensor shapes
    print("Testing DRISHYAMTLModel initialization...")
    
    # Using the primary teacher/student backbone
    model = DRISHYAMTLModel(backbone_name='tu-tf_efficientnet_b4.ns_jft_in1k', pretrained=False)
    
    # Create a dummy batch: 2 images, 3 channels, 384x384 resolution
    dummy_input = torch.randn(2, 3, 384, 384)
    
    print("Forward pass in progress...")
    logits, masks = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Logits shape (expected 2, 5): {logits.shape}")
    print(f"Masks shape (expected 2, 4, 384, 384): {masks.shape}")
    print("Success! MTL architecture is ready.")
