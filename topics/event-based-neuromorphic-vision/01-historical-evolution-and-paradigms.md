---
title: "Event-Based Vision: Historical Evolution & Paradigms"
type: evolution-guide
domain: Event-Based & Neuromorphic Vision
tags:
  - evolution
  - history
  - neuromorphic
  - dvs
  - snn
updated: 2026-09-08
aliases:
  - Neuromorphic Evolution Guide
---

# 📜 Event-Based Vision: Historical Evolution & Paradigms

## 1. The Breakdown of Frame-Based Video Acquisition
Standard camera sensors expose an array of millions of pixels simultaneously at a fixed clock rate (e.g., 30, 60, or 120 FPS). This synchronous paradigm introduces two fundamental physical limitations:
1. **Severe Motion Blur & Low Temporal Resolution**: Fast-moving objects (e.g. drone blades, high-speed projectiles, ballistic robotics) displace across dozens of pixels within a single $33\text{ ms}$ exposure window, destroying edge contours.
2. **Bandwidth & Power Waste**: A static camera records gigabytes of identical background pixels, draining batteries and memory bandwidth.

## 2. The Neuromorphic Sensor Breakthrough (DVS / DAVIS)
Bio-inspired **Dynamic Vision Sensors (DVS)**, pioneered by Tobi Delbruck and commercialized by **Prophesee / Sony**, emulate the biological retina. Each pixel operates completely autonomously:
- It responds strictly to changes in **logarithmic light intensity**:
  $$\Delta \ln(I(x, y, t)) \ge \pm C$$
- When the change exceeds threshold $C$, the pixel emits an asynchronous, microsecond-timestamped **Event Token**:
  $$e_k = (x_k, y_k, t_k, p_k)$$
  where $(x_k, y_k)$ is the pixel address, $t_k$ is the microsecond timestamp ($1\,\mu\text{s}$ resolution), and $p_k \in \{-1, +1\}$ is polarity.
- **Dynamic Range**: Exceeds **$>120\text{ dB}$** (compared to $60-70\text{ dB}$ of standard CMOS cameras), capturing objects in direct sunlight and pitch-black darkness without sensor saturation.

## 3. The Computational Evolution: From Frame Reconstruction to Native SNNs
- **First Wave (2015–2020)**: Reconstructed artificial "frames" from event accumulation buffers, running traditional CNNs. This eliminated the low-latency and low-power benefits.
- **Second Wave (2021–2024)**: Surface representations (Time Surfaces, Voxel Grids, Asynchronous Graph Neural Networks).
- **Third Wave (2025–2026 SOTA)**: **End-to-End Spiking Neural Networks (SNNs)** and **Spiking Vision Transformers (Spikformers)** running directly on neuromorphic hardware (Intel Loihi 2, SynSense, FPGA SYNtzulu) with milliwatt energy consumption.
