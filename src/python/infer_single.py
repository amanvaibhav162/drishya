import argparse
import sys
import os
import torch
import cv2
import numpy as np

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_student import DRISHYAStudentMTL

def run_inference(image_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DRISHYAStudentMTL().to(device)
    
    if os.path.exists(model_path):
        weights = torch.load(model_path, map_location=device)
        model.load_state_dict(weights, strict=False)
    model.eval()

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
        
    img = cv2.resize(img, (384, 384))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    tensor = (tensor - mean) / std

    with torch.no_grad():
        logits, _ = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        l_vals = logits[0].cpu().numpy()

    print("PROBS:" + " ".join([f"{p:.6f}" for p in probs]))
    print("LOGITS:" + " ".join([f"{l:.6f}" for l in l_vals]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to fundus image")
    parser.add_argument("--model", required=True, help="Path to student model weights")
    args = parser.parse_args()
    run_inference(args.image, args.model)
