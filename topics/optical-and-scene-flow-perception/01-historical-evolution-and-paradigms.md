---
title: "Optical & Scene Flow: Historical Evolution & Paradigms"
type: evolution-guide
domain: Optical Flow & Scene Flow Perception
tags:
  - evolution
  - history
  - optical-flow
  - scene-flow
  - raft
  - gmflow
updated: 2026-09-08
aliases:
  - Flow Evolution Guide
---

# 📜 Optical & Scene Flow: Historical Evolution & Paradigms

From differential variational formulations to deep all-pairs correlation pyramids and cross-attention matching.

Related notes: [[topics/optical-and-scene-flow-perception/00-optical-and-scene-flow-perception-moc|Optical Flow MOC]].

---

## 1. Timeline of Motion Estimation Breakthroughs

```mermaid
timeline
    title Evolution of Optical & Scene Flow
    1981 : Differential Calculus : Horn-Schunck global variational regularization & Lucas-Kanade local least squares.
    2004 : Variational Coarse-to-Fine : Brox et al. warping with spatial TV-L1 regularization for sub-pixel accuracy.
    2015 : Deep End-to-End (FlowNet) : Fischer et al. prove CNNs can regress dense displacement vectors directly.
    2020 : RAFT (Recurrent All-Pairs Correlation) : Teed & Deng introduce 4D multi-scale correlation volumes with gated recurrent updates.
    2022-2026 : Global Matching Transformers (GMFlow / UniMatch) : Cross-attention replacing recurrent GRU iterations, achieving single-pass real-time inference (>60 FPS).

```

---

## 2. Paradigms & Generational Shifts

- **1st Generation: Differential Formulations (1981–2000s)**:
  - Relied on the Brightness Constancy Assumption $I(x+u, y+v, t+1) = I(x, y, t)$. Assumed motion was infinitesimal ($\Delta x \to 0$); failed completely on rapid displacements without coarse-to-fine pyramids.
- **2nd Generation: Deep Recurrent Multi-Scale Warping (RAFT, 2020)**:
  - Extracted deep features per pixel, constructed a complete $4\text{D}$ correlation volume between all pairs of pixels, and updated flow iteratively via a ConvGRU.
- **3rd Generation: Global Attention & Unification (GMFlow / UniMatch, 2024–2026)**:
  - Formulates flow as cross-frame feature matching via self- and cross-attention transformers. Unifies stereo matching, optical flow, and depth estimation into a single weight-shared foundation model.
