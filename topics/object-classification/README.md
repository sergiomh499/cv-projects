# Object Classification Playbook

# Overview
Object Classification categorizes images, cropped regions-of-interest (RoIs), or multi-modal visual tokens into discrete predefined classes. While conceptually the simplest vision task, modern applications demand extreme efficiency, open-vocabulary generalization, robust fine-grained discrimination, and minimal out-of-distribution (OOD) failure modes.

## SOTA & Research
- **Seminal & Modern Papers**:
  - *AlexNet, VGG, ResNet* (He et al., 2015): Canonical deep residual learning stabilizing deep backbones with identity shortcuts.
  - *Vision Transformer (ViT)* (Dosovitskiy et al., 2020): Reformulated visual processing as sequence modeling over non-overlapping image patches with self-attention.
  - *Swin Transformer* (Liu et al., 2021): Shifted window self-attention yielding linear computational complexity with image scale.
  - *ConvNeXt / ConvNeXt V2* (Liu et al., 2022; Woo et al., 2023): Modernized pure ConvNets integrating Transformer design choices while maintaining convolution efficiency.
  - *CLIP & SigLIP* (Radford et al., 2021; Zhai et al., 2023): Contrastive language-image pre-training unlocking zero-shot classification via text embeddings.
- **Evaluation Benchmarks & Metrics**:
  - ImageNet-1K, ImageNet-V2, CIFAR-100, iNaturalist (fine-grained), ImageNet-O / ImageNet-A (adversarial and OOD).
  - Top-1 Accuracy, Top-5 Accuracy, Calibration Error (ECE - Expected Calibration Error).

## Architecture Alternatives & Trade-offs
| Architecture | Top-1 (ImageNet-1K) | Latency (FP16 / Edge) | Inductive Bias | Best Suited For |
| :--- | :--- | :--- | :--- | :--- |
| **Lightweight ConvNet (MobileNetV4, EfficientNet-Lite)** | 75 - 82% | <1.5 ms (CPU/Edge) | Strong translation equivariance | Mobile, microcontrollers, embedded IoT |
| **Modern Heavy ConvNet (ConvNeXt-B/L)** | 84 - 87% | 4 - 10 ms (GPU) | Moderate | High-accuracy visual classification servers |
| **Vision Transformer (ViT-H, Swin-B)** | 85 - 89% | 10 - 30 ms (GPU) | Weak (requires large-scale data pre-training) | Multimodal foundation encoders, dense context |
| **Zero-Shot VLM (CLIP / SigLIP)** | 78 - 84% (zero-shot) | 5 - 15 ms (text + image) | Open-vocabulary semantic embedding | Dynamic taxonomy, anomaly filtering |

## Popular Repos & Integrations
- **[timm (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models)**: The de facto PyTorch library for thousands of pre-trained image classification backbones.
- **[OpenCLIP](https://github.com/mlfoundations/open_clip)**: Open-source implementation and trained checkpoints of contrastive vision-language models.
- **[Torchvision Models](https://pytorch.org/vision/stable/models.html)**: Standard, heavily tested backbones maintained by the core PyTorch team.
- **Tooling Integrations**:
  - **FiftyOne**: Inspect confusion matrices, visualize t-SNE / UMAP embeddings of intermediate layers, identify mislabeled ground-truth data.
  - **Rerun**: Visualize real-time classification probability distributions as bar charts alongside live video feeds.

## End-to-End Pipeline & Workarounds
1. **Pipeline Stages**:
   - Crop / Bounding-box patch extraction -> Aspect-preserving resize -> Mean/std normalization -> Forward pass -> Softmax -> Temperature scaling / calibration.
2. **Common Traps & Edge Cases**:
   - *Overconfidence on OOD Data*: Standard softmax outputs high confidence (e.g. >95%) on completely unseen background noise.
   - *Fine-Grained Feature Loss*: Severe downsampling (e.g. 224x224 standard) strips discriminative textures on small components.
3. **Engineering Workarounds**:
   - **Temperature Scaling & Platt Scaling**: Re-calibrate logit outputs post-hoc on a validation split to ensure probabilities match empirical accuracy.
   - **Mahalanobis Distance / Deep Feature Anomaly Scores**: Extract penultimate layer embeddings and reject inputs exceeding distance thresholds from class centroids.
   - **Test-Time Augmentation (TTA)**: Average predictions across original and flipped/scaled variants for critical low-confidence decisions.

## Deployment & Real-time Notes
- **Hardware Acceleration**:
  - Classification backbones are compute-dense and easily reach maximum hardware utilization in TensorRT, ONNX Runtime, and OpenVINO.
- **Batched RoI Classification**:
  - When pairing classification as a secondary stage after an object detector, batch RoI crops into fixed tensor chunks (e.g., batch size 16 or 32) to saturate GPU streaming multiprocessors.
- **Weight Quantization**:
  - Modern ConvNets (MobileNetV4, ResNet) easily support INT8 PTQ (Post-Training Quantization) with <0.5% Top-1 drop, making them ideal for edge execution.
