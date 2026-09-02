"""
DRISHYA XAI Module
==================
Explainable AI tools for Diabetic Retinopathy screening.
Provides GradCAM/GradCAM++ heatmap generation, inference, and visualization
for the DRISHYAMTLModel (EfficientNet-B4 teacher).
"""

from .gradcam import generate_heatmap, ClassificationWrapper, get_target_layer, create_fundus_fov_mask
from .inference import load_teacher_model, preprocess_image, run_single_image
from .visualization import overlay_heatmap, create_comparison_panel, save_outputs

__all__ = [
    "generate_heatmap",
    "ClassificationWrapper",
    "get_target_layer",
    "create_fundus_fov_mask",
    "load_teacher_model",
    "preprocess_image",
    "run_single_image",
    "overlay_heatmap",
    "create_comparison_panel",
    "save_outputs",
]
