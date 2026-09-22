import os
import cv2
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from model_student import DRISHYAStudentMTL

class ClassificationWrapper(nn.Module):
    """Wraps the MTL model to only return classification logits for Grad-CAM"""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        logits, _ = self.model(x)
        return logits

def calculate_biomarkers(masks):
    """
    Computes clinical biomarkers from the 4 segmentation masks.
    Mask 0: Microaneurysms
    Mask 1: Exudates
    Mask 2: Hemorrhages
    Mask 3: Vessels
    """
    probs = torch.sigmoid(masks).cpu().numpy()[0] # (4, 384, 384)
    
    # 1. Microaneurysm Count (Mask 0)
    ma_thresh = (probs[0] > 0.5).astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(ma_thresh)
    num_ma = num_labels - 1 # subtract background
    
    # 2. Exudate Area Percentage (Mask 1)
    ex_thresh = (probs[1] > 0.5).astype(np.uint8)
    exudate_area = np.sum(ex_thresh)
    total_area = 384 * 384
    ex_pct = (exudate_area / total_area) * 100.0
    
    # 3. Hemorrhage Quadrants (Mask 2)
    he_thresh = (probs[2] > 0.5).astype(np.uint8)
    h, w = 384, 384
    quads = [
        he_thresh[0:h//2, 0:w//2],
        he_thresh[0:h//2, w//2:w],
        he_thresh[h//2:h, 0:w//2],
        he_thresh[h//2:h, w//2:w]
    ]
    quad_count = sum([1 for q in quads if np.sum(q) > 10]) # At least 10 pixels to count as present in quadrant
    
    return num_ma, ex_pct, quad_count

def main():
    print("=========================================================")
    print("  DRISHYA: Inference, Grad-CAM & Export Pipeline")
    print("=========================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Model
    model = DRISHYAStudentMTL().to(device)
    model_path = "models/student_mtl_lcnet.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✔ Loaded trained weights from {model_path}")
    else:
        print(f"⚠ WARNING: {model_path} not found. Using random weights!")
        
    model.eval()
    
    # 1. ONNX EXPORT
    print("\n[1/3] Exporting model to ONNX...")
    dummy_input = torch.randn(1, 3, 384, 384).to(device)
    onnx_path = "dr_mtl_student_b4.onnx"
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        input_names=["input_image"], 
        output_names=["dr_logits", "segmentation_masks"],
        dynamic_axes={"input_image": {0: "batch_size"}, "dr_logits": {0: "batch_size"}, "segmentation_masks": {0: "batch_size"}},
        opset_version=12
    )
    print(f"✔ ONNX Export complete: {onnx_path}")
    
    # Setup for Inference & GradCAM
    os.makedirs("output/gradcam", exist_ok=True)
    wrapper = ClassificationWrapper(model)
    target_layers = [model.encoder] 
    
    try:
        cam = GradCAMPlusPlus(model=wrapper, target_layers=target_layers, use_cuda=(device.type=='cuda'))
    except Exception as e:
        print(f"Warning: GradCAM init failed with generic target layer. Exception: {e}")
        cam = None
    
    # 2. RUN INFERENCE ON SAMPLE IMAGES
    print("\n[2/3] Running Inference & Extracting Biomarkers...")
    data_dir = "data/train_images/"
    if not os.path.exists(data_dir):
        print(f"⚠ WARNING: No test images found in {data_dir}. Skipping CSV/GradCAM generation.")
        return
        
    image_files = [f for f in os.listdir(data_dir) if f.endswith('.png') or f.endswith('.jpg')][:5] # Test on 5 images
    if not image_files:
        print(f"⚠ WARNING: {data_dir} is empty.")
        return
        
    csv_data = []
    
    for img_name in image_files:
        img_path = os.path.join(data_dir, img_name)
        img_cv2 = cv2.imread(img_path)
        img_cv2 = cv2.resize(img_cv2, (384, 384))
        rgb_img = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB) / 255.0
        
        # Preprocess for model
        input_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)
        input_tensor = (input_tensor - mean) / std
        
        # Inference
        with torch.no_grad():
            logits, masks = model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred_class = torch.argmax(probs).item()
            conf = probs[pred_class].item() * 100.0
            
        referable = "YES" if pred_class >= 2 else "NO"
        num_ma, ex_pct, quad_count = calculate_biomarkers(masks)
        
        csv_data.append({
            "image_name": img_name,
            "predicted_grade": pred_class,
            "confidence_pct": f"{conf:.1f}%",
            "referable_dr": referable,
            "num_microaneurysms": num_ma,
            "exudate_area_pct": f"{ex_pct:.2f}%",
            "hemorrhage_quadrants": quad_count
        })
        
        # 3. GRAD-CAM++
        if cam is not None:
            # We target the predicted class
            targets = [ClassifierOutputTarget(pred_class)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            
            # Overlay
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            cv2.imwrite(f"output/gradcam/{img_name}_gradcam.png", cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
            
    # Save CSV
    df = pd.DataFrame(csv_data)
    df.to_csv("output/lesion_features.csv", index=False)
    print("✔ Biomarker CSV generated at output/lesion_features.csv")
    print("✔ Grad-CAM images saved to output/gradcam/")
    
    print("\n=========================================================")
    print("🎉 ALL DELIVERABLES SUCCESSFULLY EXPORTED 🎉")
    print("=========================================================")

if __name__ == "__main__":
    main()
