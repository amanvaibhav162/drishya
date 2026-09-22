import os
import time
import pytest
import numpy as np
import onnxruntime as ort

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def test_onnx_model_file_exists():
    onnx_path = os.path.join(BASE_DIR, "models", "drishya_student_lcnet.onnx")
    assert os.path.exists(onnx_path), f"ONNX file not found: {onnx_path}"

def test_onnx_inference_session():
    onnx_path = os.path.join(BASE_DIR, "models", "drishya_student_lcnet.onnx")
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    input_meta = session.get_inputs()[0]
    assert input_meta.name == "input_fundus"
    assert input_meta.shape in [[1, 3, 384, 384], ['batch_size', 3, 384, 384], [-1, 3, 384, 384], [1, 3, -1, -1]]

    output_names = [o.name for o in session.get_outputs()]
    assert "dr_logits" in output_names
    assert "lesion_masks" in output_names

def test_onnx_forward_pass_and_latency():
    onnx_path = os.path.join(BASE_DIR, "models", "drishya_student_lcnet.onnx")
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    dummy_input = np.random.randn(1, 3, 384, 384).astype(np.float32)
    
    # Warmup
    _ = session.run(None, {"input_fundus": dummy_input})
    
    # Benchmark 5 runs
    start = time.perf_counter()
    for _ in range(5):
        outputs = session.run(None, {"input_fundus": dummy_input})
    avg_latency_ms = ((time.perf_counter() - start) / 5) * 1000.0
    
    logits, masks = outputs[0], outputs[1]
    assert logits.shape == (1, 5), f"Expected logits (1, 5), got {logits.shape}"
    assert masks.shape == (1, 4, 384, 384), f"Expected masks (1, 4, 384, 384), got {masks.shape}"
    
    # Check probabilities
    exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
    assert np.isclose(np.sum(probs), 1.0, atol=1e-4)
    
    print(f"\n[ONNX Benchmark] Average CPU Latency: {avg_latency_ms:.2f} ms per frame")
    # Edge requirement: < 250ms on modern CPU
    assert avg_latency_ms < 250.0
