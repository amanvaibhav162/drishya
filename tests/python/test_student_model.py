import os
import sys
import pytest
import torch
import numpy as np

# Add src/python to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(BASE_DIR, "src", "python"))

from model_student import DRISHYAStudentMTL

def test_model_instantiation():
    model = DRISHYAStudentMTL(pretrained=False)
    assert model is not None
    assert hasattr(model, 'unet')
    assert hasattr(model, 'classifier')
    assert hasattr(model, 'msag')

def test_forward_pass_dimensions():
    model = DRISHYAStudentMTL(pretrained=False)
    model.eval()
    
    dummy_input = torch.randn(2, 3, 384, 384)
    with torch.no_grad():
        logits, masks = model(dummy_input)
        
    assert logits.shape == (2, 5), f"Expected logits (2, 5), got {logits.shape}"
    assert masks.shape == (2, 4, 384, 384), f"Expected masks (2, 4, 384, 384), got {masks.shape}"

def test_weights_loading():
    weights_path = os.path.join(BASE_DIR, "models", "student_mtl_lcnet_best.pth")
    assert os.path.exists(weights_path), f"Checkpoint missing: {weights_path}"
    
    model = DRISHYAStudentMTL(pretrained=False)
    weights = torch.load(weights_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(weights, strict=False)
    
    # Ensure backbone weights matched
    assert len(missing) == 0, f"Critical weights missing: {missing}"

def test_softmax_probabilities():
    model = DRISHYAStudentMTL(pretrained=False)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 384, 384)
    with torch.no_grad():
        logits, _ = model(dummy_input)
        probs = torch.softmax(logits, dim=1).numpy()
        
    assert np.isclose(np.sum(probs), 1.0, atol=1e-4)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
