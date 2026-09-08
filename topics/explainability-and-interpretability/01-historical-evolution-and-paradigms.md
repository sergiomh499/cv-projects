---
title: "Explainability & Interpretability: Historical Evolution & Paradigms"
type: evolution-guide
domain: Explainability & Interpretability
tags:
  - evolution
  - history
  - xai
  - grad-cam
  - concept-bottlenecks
  - mechanistic-interpretability
updated: 2026-09-08
aliases:
  - Explainability Evolution
---

# 📜 Explainability & Interpretability: Historical Evolution & Paradigms

## 1. The Era of Post-Hoc Saliency (2014–2020)
The early phase of explainable computer vision relied on gradient backpropagation to attribute predictions to input pixels:
- **Vanilla Gradients & Guided Backpropagation**: Visualized fine-grained pixel highlights but frequently failed sanity checks (behaving essentially as edge detectors).
- **Grad-CAM (Gradient-Weighted Class Activation Mapping)**: Weighted penultimate convolutional feature maps by pooled gradients, creating human-friendly heatmaps highlighting object locations.

## 2. The Faithfulness Crisis & Concept Bottlenecks (2020–2024)
Empirical investigations revealed that post-hoc saliency maps were often unfaithful: modifying random model weights had negligible impact on generated heatmaps.

To solve this, **Concept Bottleneck Models (CBMs)** emerged (Koh et al.), modifying the network architecture:
$$x \longrightarrow c \in \mathbb{R}^k \longrightarrow \hat{y}$$
The model is forced to explicitly predict predefined, human-interpretable concepts (e.g., *has stripes*, *has wings*) before computing the final class logits.

## 3. Mechanistic Interpretability & Sparse Autoencoders (2025–2026)
Modern foundation vision models (ViTs, VLMs) contain millions of polysemantic neurons (a single neuron responding to cars, faces, and text simultaneously).
**Mechanistic Interpretability** applies **Sparse Autoencoders (SAEs)** to disentangle polysemantic latent activations into thousands of sparse, monosemantic feature vectors, enabling exact circuit tracing of how spatial tokens are bound together.
