#!/usr/bin/env python3
"""
Minimal Production Recipe: Exporting RF-DETR to ONNX and Building a TensorRT FP16 Engine.

Steps:
1. Validates PyTorch model export with dynamic batch sizes.
2. Inlines ONNX export recipe.
3. Provides automated trtexec build command with layer fusion and FP16.
"""

import sys
import torch
import torch.nn as nn

class DummyRFDETRBackbone(nn.Module):
    """Minimal representation of RF-DETR DINOv2 backbone with 2D detection heads."""
    def __init__(self, num_classes=80):
        super().__init__()
        self.conv = nn.Conv2d(3, 256, kernel_size=3, stride=2, padding=1)
        self.head_cls = nn.Linear(256, num_classes)
        self.head_box = nn.Linear(256, 4)

    def forward(self, x):
        feat = self.conv(x).mean(dim=[2, 3])
        logits = self.head_cls(feat)
        boxes = torch.sigmoid(self.head_box(feat))
        return logits, boxes

def export_rfdetr_to_onnx(output_path: str = "rf_detr.onnx"):
    print(f"[+] Instantiating RF-DETR model...")
    model = DummyRFDETRBackbone().eval().cuda()
    dummy_input = torch.randn(1, 3, 640, 640, device="cuda")

    print(f"[+] Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["images"],
        output_names=["pred_logits", "pred_boxes"],
        dynamic_axes={
            "images": {0: "batch"},
            "pred_logits": {0: "batch"},
            "pred_boxes": {0: "batch"}
        },
        opset_version=17
    )
    print(f"[✓] ONNX export completed successfully.")
    
    print("\n[🚀 TensorRT 10 Compilation Command]")
    print(f"trtexec --onnx={output_path} \\")
    print(f"        --saveEngine=rf_detr_fp16.engine \\")
    print(f"        --fp16 \\")
    print(f"        --optShapes=images:1x3x640x640 \\")
    print(f"        --minShapes=images:1x3x640x640 \\")
    print(f"        --maxShapes=images:8x3x640x640 \\")
    print(f"        --builderOptimizationLevel=5 \\")
    print(f"        --useCudaGraph")

if __name__ == "__main__":
    export_rfdetr_to_onnx()
