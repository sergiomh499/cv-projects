---
title: "D-FINE: Redefine Regression Task for Real-Time Object Detection"
architecture_class: "Real-Time Object Detection"
primary_license: "Apache-2.0"
domain: "Real-Time Detectors & Segmenters"
status: evergreen
updated: 2026-08-15
tags:
  - architecture
  - object-detection
  - real-time-perception
  - vision-transformer
  - d-fine
  - nms-free
  - fine-grained-distribution-refinement
  - non-uniform-distribution
  - global-local-weighting
  - edge-inference
aliases:
  - "D-FINE"
  - "DFINE"
  - "Redefine Regression DETR"
  - "Fine-Grained Distribution Refinement"
---

# ⚡ D-FINE: Redefine Regression Task for Real-Time Object Detection

## 1. Executive Brief & Significance

In real-time object detection, bounding box regression has historically followed two rigid paradigms:
1. **Direct Coordinate Regression**: Models (such as Faster R-CNN, [[architectures/real-time-detectors-and-segmenters/yolox|YOLOX]], and early DETRs) directly predict continuous 4D coordinates $(x, y, w, h)$. While computationally minimal, direct regression struggles to capture localization uncertainty in the presence of severe occlusion, motion blur, and ambiguous object boundaries.
2. **Uniform Distribution Focal Loss (DFL)**: Models (such as [[architectures/real-time-detectors-and-segmenters/yolov8|YOLOv8]], [[architectures/real-time-detectors-and-segmenters/yolov9|YOLOv9]], and [[architectures/real-time-detectors-and-segmenters/yolo11|YOLO11]]) discretize coordinates into uniform integer bins ($y \in \{0, 1, \dots, 15\}$). However, uniform bin discretization introduces rigid quantization errors and scales quadratically in parameter count when higher localization granularity is required.

```mermaid
flowchart TD
    subgraph InputStage ["1. Vision Backbone (Lightweight HGNetv2)"]
        ImgIn["Input Image (3 x 640 x 640)"] --> Stem["Stem & Downsampling Stages"]
        Stem --> S3["S3 (Stride 8: 80x80)"]
        Stem --> S4["S4 (Stride 16: 40x40)"]
        Stem --> S5["S5 (Stride 32: 20x20)"]
    end

    subgraph HybridEncoder ["2. Lightweight Hybrid Encoder"]
        S5 --> AIFI["AIFI (Single-Scale Attention on S5)"]
        AIFI --> CCFM["Cross-Scale Conv Feature Fusion Module (CCFM)"]
        S4 --> CCFM
        S3 --> CCFM
        CCFM --> MemoryFeats["Multi-Scale Feature Memory Maps"]
    end

    subgraph DecoderStage ["3. Iterative FDR Transformer Decoder (L=3 Layers)"]
        MemoryFeats --> Layer1["FDR Decoder Layer 1: Initial Coarse Distribution"]
        Layer1 --> GOL1["Global-Local Weighting (GO-L)"]
        GOL1 --> Layer2["FDR Decoder Layer 2: Residual Distribution Refinement"]
        Layer2 --> GOL2["Global-Local Weighting (GO-L)"]
        GOL2 --> Layer3["FDR Decoder Layer 3: Sub-Pixel Boundary Convergence"]
    end

    subgraph OutputStage ["4. Deterministic Set Prediction"]
        Layer3 --> FinalBoxes["Sub-Pixel Accurate Bounding Boxes (NMS-Free, TensorRT < 2ms)"]
        Layer3 --> FinalScores["Calibrated Classification Probabilities"]
    end
```

**D-FINE** (Peng et al., ICLR 2025 / arXiv:2410.13842) completely redefines the localization objective by introducing three architectural breakthroughs:
1. **Fine-Grained Distribution Refinement (FDR)**: Replaces fixed coordinate predictions with recursive, iterative distribution refinement. Across decoder layers, the model does not predict absolute boxes from scratch; instead, each layer predicts a residual probability distribution that refines the preceding layer's bounding box expectation.
2. **Non-Uniform Bounding Box Distribution**: Allocates high-density distribution bins near the expected object boundary and sparsely distributes bins in distant regions. This non-uniform spacing slashes boundary discretization error by $> 60\%$ while reducing the required bin count from 16 to 8.
3. **Global-Local Weighting (GO-L)**: Dynamically balances global semantic context with local boundary edge gradients, decoupling localization confidence from raw classification probability.

---

## 2. Component-by-Component Decomposition

| Architectural Component | Technical Specification | D-FINE-S Configuration | D-FINE-X Configuration | Parameter Distribution (%) | Latency / Compute Distribution (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Architectural Paradigm** | Fine-Grained Distribution Refinement DETR | HGNetv2 + FDR Transformer Decoder | HGNetv2 + FDR Transformer Decoder | $100\%$ Total ($10.4\text{M}$ / $62.8\text{M}$) | $100\%$ Total ($24.8\text{G}$ / $202.0\text{G}$) |
| **Vision Backbone** | **Lightweight HGNetv2** | 5 stages (S1-S5), channels $[64, 128, 256, 512]$ | 5 stages (S1-S5), channels $[64, 128, 256, 512, 1024]$ | $\sim 53.5\%$ ($5.6\text{M}$ / $33.6\text{M}$) | $\approx 58.0\%$ forward latency |
| **AIFI & CCFM Encoder** | Single-Scale Self-Attention + RepConv Fusion | $D=256$, 8 heads on S5 ($20\times 20$); CCFM on S3, S4, S5 | $D=384$, 8 heads on S5 ($20\times 20$); CCFM on S3, S4, S5 | $\sim 26.5\%$ ($2.8\text{M}$ / $16.6\text{M}$) | $\approx 25.0\%$ forward latency |
| **FDR Transformer Decoder** | Iterative Residual Distribution Layers | $L=3$ layers (rapid convergence), $Q=300, D=256$ | $L=4$ layers, $Q=300, D=384$ | $\sim 16.2\%$ ($1.7\text{M}$ / $10.2\text{M}$) | $\approx 14.5\%$ forward latency |
| **Non-Uniform Reg Head** | Non-Uniform Bin Projection ($N=8$ Bins) | Decoupled 3-layer MLP projecting to $4 \times 8$ non-uniform bins | Decoupled 3-layer MLP projecting to $4 \times 8$ non-uniform bins | $\sim 2.5\%$ ($0.2\text{M}$ / $1.6\text{M}$) | $\approx 1.5\%$ forward latency |
| **GO-L Module** | Global-Local Uncertainty Weighting | Channel-wise uncertainty estimation + Local gradient gate | Channel-wise uncertainty estimation + Local gradient gate | $\sim 1.3\%$ ($0.1\text{M}$ / $0.8\text{M}$) | $\approx 1.0\%$ forward latency |

```mermaid
flowchart LR
    subgraph NonUniformBins ["Uniform DFL vs Non-Uniform FDR Bin Allocation"]
        subgraph UniformDFL ["Uniform Bins (YOLOv8 / DFL): 16 Uniform Steps"]
            U1["0"] --- U2["1"] --- U3["2"] --- U4["..."] --- U5["14"] --- U6["15"]
        end

        subgraph NonUniformFDR ["D-FINE Non-Uniform Bins: Dense at Center, Sparse at Extremes"]
            N1["-4.0 (Sparse)"] --- N2["-1.5"] --- N3["-0.5 (Dense)"] --- N4["0.0"] --- N5["+0.5 (Dense)"] --- N6["+1.5"] --- N7["+4.0 (Sparse)"]
        end
    end
```

---

## 3. Mathematical Formulations & Loss Functions

### A. Non-Uniform Distribution Bin Formulation
Standard DFL allocates uniform discrete bins $b_i = i$ for $i \in \{0, 1, \dots, N-1\}$.

D-FINE defines a **non-uniform power-law partition function** over $N$ bins ($N=8$, indexed $i \in \{0, \dots, N-1\}$):

$$b_i = \text{sign}\left( i - \frac{N-1}{2} \right) \cdot \left| \frac{i - \frac{N-1}{2}}{\frac{N-1}{2}} \right|^\gamma \cdot R_{\text{max}}$$

where $\gamma > 1.0$ (typically $\gamma = 1.6$) concentrates discrete intervals tightly around the expected boundary residual zero ($0.0$), and $R_{\text{max}}$ defines the maximum residual search radius.

```mermaid
flowchart TD
    subgraph NonUniformMath ["Non-Uniform Bin Interval Distribution"]
        Index["Bin Index i in {0, 1, ..., 7}"] --> PowerLaw["Apply Power-Law Transformation: b_i = sign(i - 3.5) * |(i - 3.5)/3.5|^1.6 * R_max"]
        PowerLaw --> Densities["Generates Dense Sampling in [-0.5, +0.5] and Sparse Sampling in [-R_max, +R_max]"]
    end
```

### B. Fine-Grained Distribution Refinement (FDR)
Let $e^{(l-1)} \in \mathbb{R}^4$ denote the decoded bounding box coordinates at decoder layer $l-1$. Rather than predicting an entirely new box, decoder layer $l$ predicts a discrete probability distribution $\mathbf{P}^{(l)} \in \mathbb{R}^{4 \times N}$ over the non-uniform bins $\mathbf{b}$:

$$\mathbf{P}_j^{(l)} = \text{Softmax}\left( \mathbf{z}_j^{(l)} \right), \quad j \in \{l, t, r, b\}$$

The residual refinement offset $\Delta e_j^{(l)}$ is computed as the statistical expectation over the non-uniform bin vector $\mathbf{b}$:

$$\Delta e_j^{(l)} = \sum_{i=0}^{N-1} b_i \cdot P_{j, i}^{(l)}$$

The updated bounding box coordinate at layer $l$ is:

$$e_j^{(l)} = e_j^{(l-1)} + \Delta e_j^{(l)} \cdot s^{(l)}$$

where $s^{(l)}$ is a layer-dependent scale factor.

### C. Global-Local Weighting (GO-L)
To prevent misleading classification scores from dominating localization targets, GO-L calculates a combined localization weight $\mathcal{W}_{\text{GO-L}}$:

$$\mathcal{W}_{\text{GO-L}} = \sigma \left( \mathbf{W}_g \cdot \mathbf{f}_{\text{global}} \right) \odot \exp \left( - \frac{\text{Var}(\mathbf{P}^{(l)})}{\tau} \right)$$

where $\text{Var}(\mathbf{P}^{(l)}) = \sum_{i=0}^{N-1} P_{j, i}^{(l)} (b_i - \Delta e_j^{(l)})^2$ measures distribution dispersion (entropy). Low dispersion indicates sharp, unambiguous boundary edges.

### D. Multi-Task Training Objective
The complete D-FINE objective optimized via Hungarian bipartite matching is:

$$\mathcal{L}_{\text{total}} = \sum_{l=1}^L \left( \lambda_{\text{cls}} \mathcal{L}_{\text{VFL}}(p^{(l)}, \hat{t}) + \lambda_{\text{fdr}} \mathcal{W}_{\text{GO-L}} \cdot \mathcal{L}_{\text{FDR}}(\mathbf{P}^{(l)}, \hat{\mathbf{P}}) + \lambda_{\text{giou}} \mathcal{L}_{\text{GIoU}}(b^{(l)}, \hat{b}) \right)$$

where $\mathcal{L}_{\text{FDR}}$ is the soft cross-entropy between the predicted non-uniform distribution $\mathbf{P}^{(l)}$ and the continuous ground-truth boundary projection:

$$\mathcal{L}_{\text{FDR}}(\mathbf{P}, \hat{\mathbf{P}}) = - \sum_{j=1}^4 \left( (b_{k+1} - y_j) \log(P_{j, k}) + (y_j - b_k) \log(P_{j, k+1}) \right)$$

---

## 4. Benchmark Evaluation & Performance Profiles

The following performance matrix benchmarks D-FINE across standard COCO 2017 test-dev / minival:

| Architecture | Model Variant | Parameters (M) | FLOPs (G) | mAP 50:95 (%) | mAP 50 (%) | T4 TRT FP16 (ms) | RTX 4090 FP16 (ms) | Orin AGX FP16 (ms) | Orin AGX INT8 (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **D-FINE-N** | Nano | 3.8M | 9.4G | 42.8% | 58.8% | 1.80 ms | 0.68 ms | 3.6 ms | 1.9 ms |
| **D-FINE-S** | Small | 10.4M | 24.8G | 48.5% | 65.8% | 2.50 ms | 0.95 ms | 5.8 ms | 2.9 ms |
| **D-FINE-M** | Medium | 19.4M | 56.2G | 52.1% | 69.8% | 4.20 ms | 1.55 ms | 9.8 ms | 4.9 ms |
| **D-FINE-L** | Large | 31.2M | 92.4G | 54.0% | 72.0% | 6.10 ms | 2.30 ms | 14.5 ms | 7.2 ms |
| **D-FINE-X** | X-Large | 62.8M | 202.0G | 55.8% | 74.2% | 11.20 ms | 4.20 ms | 26.5 ms | 13.0 ms |

*Measurements taken at $640 \times 640$ resolution, batch size 1, using TensorRT 10. Direct NMS-free end-to-end latency.*

---

## 5. Edge Deployment, TensorRT Optimization & Gotchas

### A. Non-Uniform Distribution Matrix Multiplication Fusion
In D-FINE, the residual coordinate expectation is computed via an inner product between the Softmax probability vector $\mathbf{P} \in \mathbb{R}^{B \times Q \times 4 \times 8}$ and the precomputed constant non-uniform bin vector $\mathbf{b} \in \mathbb{R}^{8}$:

$$\Delta e = \mathbf{P} \cdot \mathbf{b}^T$$

**Optimization Recipe**:
- Export the non-uniform bin values $\mathbf{b}$ as a constant tensor inside the ONNX graph.
- TensorRT automatically fuses the Softmax and MatMul operations into a single systolic array kernel, executing in $< 0.05\text{ ms}$.

```mermaid
flowchart LR
    subgraph NonUniformFusion ["TensorRT Fused FDR Kernel"]
        Logits["FDR Logits: (B, Q, 4, 8)"] --> SoftmaxNode["Softmax along Bin Dim"]
        SoftmaxNode --> MatMulNode["GEMM with Constant Non-Uniform Bins b"]
        MatMulNode --> DecodedBoxes["Refined Box Offsets: (B, Q, 4)"]
    end
```

### B. End-to-End ONNX Export & TensorRT Compilation Recipe

```python
import torch

# 1. Load pre-trained D-FINE model
# Official repo: https://github.com/Peterande/D-FINE
model = torch.hub.load("Peterande/D-FINE", "dfine_s", pretrained=True)
model.eval()

# 2. Export directly to ONNX (NMS-free end-to-end set prediction)
dummy_input = torch.randn(1, 3, 640, 640)
torch.onnx.export(
    model,
    dummy_input,
    "dfine_s.onnx",
    input_names=["images"],
    output_names=["boxes", "scores"],
    opset_version=17,
    do_constant_folding=True
)
print("Exported D-FINE ONNX model successfully.")
```

```bash
# Build High-Throughput INT8 TensorRT Engine
trtexec \
  --onnx=dfine_s.onnx \
  --saveEngine=dfine_s_int8.engine \
  --int8 \
  --fp16 \
  --calib=dfine_calib.cache \
  --builderOptimizationLevel=5 \
  --avgRuns=100
```

---

## 6. Complete Runnable Python Blueprint

The following standalone PyTorch script implements the core innovations of D-FINE: **`NonUniformDistributionDecoder`**, **`FineGrainedDistributionRefinement` (FDR)**, and the **`GlobalLocalWeighting`** module.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NonUniformDistributionDecoder(nn.Module):
    """Generates non-uniform power-law distribution bins and computes expectation."""
    def __init__(self, num_bins=8, gamma=1.6, r_max=4.0):
        super().__init__()
        self.num_bins = num_bins
        # Precompute non-uniform bin centers: b_i = sign(i - mid) * |(i - mid)/mid|^gamma * r_max
        mid = (num_bins - 1) / 2.0
        indices = torch.arange(num_bins, dtype=torch.float32)
        norm_indices = (indices - mid) / mid
        bins = torch.sign(norm_indices) * (torch.abs(norm_indices) ** gamma) * r_max
        self.register_buffer("bins", bins.view(1, 1, 1, num_bins))  # [1, 1, 1, 8]

    def forward(self, distribution_logits):
        # distribution_logits: [B, Q, 4, num_bins]
        probs = F.softmax(distribution_logits, dim=-1)
        # Compute expectation: sum(prob_i * bin_i)
        residual_offsets = (probs * self.bins).sum(dim=-1)  # [B, Q, 4]
        return residual_offsets, probs


class GlobalLocalWeighting(nn.Module):
    """Global-Local Weighting (GO-L) module for boundary localization confidence."""
    def __init__(self, d_model=256):
        super().__init__()
        self.global_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 4, 4),
            nn.Sigmoid()
        )

    def forward(self, query_feats, distribution_probs, bins):
        # query_feats: [B, Q, d_model]
        # distribution_probs: [B, Q, 4, num_bins]
        # 1. Global semantic weighting
        w_global = self.global_gate(query_feats)  # [B, Q, 4]
        
        # 2. Local distribution dispersion (Variance / Entropy)
        expected_val = (distribution_probs * bins).sum(dim=-1, keepdim=True)
        variance = (distribution_probs * ((bins - expected_val) ** 2)).sum(dim=-1)  # [B, Q, 4]
        w_local = torch.exp(-variance / 2.0)
        
        return w_global * w_local


class FineGrainedDistributionRefinement(nn.Module):
    """Iterative FDR Transformer Decoder Layer."""
    def __init__(self, d_model=256, num_bins=8):
        super().__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8, batch_first=True)
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 4 * num_bins)
        )
        self.cls_head = nn.Linear(d_model, 80)
        self.distribution_decoder = NonUniformDistributionDecoder(num_bins=num_bins)
        self.gol = GlobalLocalWeighting(d_model=d_model)

    def forward(self, query, memory, prev_boxes):
        # query: [B, Q, d_model]
        # memory: [B, Total_Tokens, d_model]
        # prev_boxes: [B, Q, 4]
        query_refined = self.decoder_layer(query, memory)
        
        B, Q, _ = query.shape
        reg_logits = self.reg_head(query_refined).reshape(B, Q, 4, 8)
        cls_logits = self.cls_head(query_refined)
        
        # Compute residual refinement offsets
        delta_boxes, probs = self.distribution_decoder(reg_logits)
        updated_boxes = prev_boxes + delta_boxes
        
        # Compute GO-L weighting
        weights = self.gol(query_refined, probs, self.distribution_decoder.bins)
        
        return query_refined, updated_boxes, cls_logits, weights


# Smoke validation pass
if __name__ == "__main__":
    B, Q, C, N_tokens = 2, 300, 256, 8400
    query = torch.randn(B, Q, C)
    memory = torch.randn(B, N_tokens, C)
    init_boxes = torch.rand(B, Q, 4)
    
    fdr_layer = FineGrainedDistributionRefinement(d_model=256, num_bins=8)
    q_out, boxes_out, cls_out, gol_weights = fdr_layer(query, memory, init_boxes)
    
    assert q_out.shape == (B, Q, C)
    assert boxes_out.shape == (B, Q, 4)
    assert cls_out.shape == (B, Q, 80)
    assert gol_weights.shape == (B, Q, 4)
    
    print(f"Non-Uniform Bins Precomputed: {fdr_layer.distribution_decoder.bins.squeeze()}")
    print("D-FINE Blueprint validation completed successfully.")
```

---

## 7. Peer Comparisons & Cross-Links

| Architectural Dimension | D-FINE | [[architectures/real-time-detectors-and-segmenters/rt-detr|RT-DETR]] | [[architectures/real-time-detectors-and-segmenters/rt-detr-v2|RT-DETR v2]] | [[architectures/real-time-detectors-and-segmenters/yolov10|YOLOv10]] | [[architectures/real-time-detectors-and-segmenters/yolo11|YOLO11]] | [[architectures/real-time-detectors-and-segmenters/yolov12|YOLOv12]] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Localization Paradigm**| **FDR (Iterative Non-Uniform)** | Direct Bbox MLP | Discrete Sampling MLP | DFL (Uniform Bins) | DFL (Uniform Bins) | DFL + Area Attention |
| **Bin Discretization** | **Non-Uniform Power-Law ($N=8$)**| None (Continuous) | None (Continuous) | Uniform ($N=16$) | Uniform ($N=16$) | Uniform ($N=16$) |
| **Weighting Metric** | **Global-Local (GO-L)** | Joint Uncertainty | Normalized Wasserstein | Task-Aligned | Task-Aligned | Task-Aligned |
| **Post-Processing** | **NMS-Free** ($\mathcal{O}(1)$) | **NMS-Free** ($\mathcal{O}(1)$) | **NMS-Free** ($\mathcal{O}(1)$) | **NMS-Free** ($\mathcal{O}(1)$) | Heuristic NMS | Standard / NMS-Free |
| **COCO mAP-50:95 (S)** | **$48.5\%$** | $46.5\%$ | $47.8\%$ | $46.3\%$ | $47.0\%$ | $48.2\%$ |
| **COCO mAP-50:95 (X)** | **$55.8\%$** | $54.8\%$ | $55.3\%$ | $54.4\%$ | $54.7\%$ | $55.1\%$ |

- **Related Topics & MOCs**:
  - [[topics/object-detection/00-object-detection-moc|Object Detection MOC]]
  - [[topics/real-time-systems/00-real-time-systems-moc|Real-Time Systems MOC]]
  - [[topics/gpu-deployment/00-gpu-deployment-moc|GPU Deployment MOC]]
  - [[architectures/real-time-detectors-and-segmenters/rt-detr|RT-DETR: Real-Time Detection Transformer]]
  - [[architectures/real-time-detectors-and-segmenters/rt-detr-v2|RT-DETR v2: Real-Time Detection Transformer]]
  - [[architectures/real-time-detectors-and-segmenters/rf-detr|RF-DETR: NAS-Optimized Detection Transformer]]
  - [[architectures/real-time-detectors-and-segmenters/yolov10|YOLOv10: Consistent Dual Assignments]]
  - [[architectures/real-time-detectors-and-segmenters/yolov12|YOLOv12: Attention-Centric Real-Time Detection]]
