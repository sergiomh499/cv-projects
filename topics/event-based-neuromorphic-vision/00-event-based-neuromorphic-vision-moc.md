---
title: "Event-Based & Neuromorphic Vision MOC"
type: MOC
domain: Event-Based & Neuromorphic Vision
tags:
  - moc
  - computer-vision
  - event-cameras
  - neuromorphic-vision
  - prophesee
  - dvs
  - snn
status: evergreen
updated: 2026-09-08
aliases:
  - Neuromorphic Vision MOC
  - Event Vision Hub
---

# ⚡ Event-Based & Neuromorphic Vision MOC

> **Navigation**: [[README|🏠 Central Knowledge Hub]] / **Event-Based & Neuromorphic Vision
tags:
  - moc
  - computer-vision
  - event-cameras
  - neuromorphic-vision
  - prophesee
  - dvs
  - snn
status: evergreen
updated: 2026-09-08
aliases:
  - Neuromorphic Vision MOC
  - Event Vision Hub
---

# ⚡ Event-Based & Neuromorphic Vision MOC

Welcome to the **Event-Based & Neuromorphic Vision** knowledge domain. This hub covers bio-inspired asynchronous Dynamic Vision Sensors (DVS), microsecond temporal resolution, high dynamic range ($>120\text{ dB}$), Spiking Neural Networks (SNNs), and event-based SLAM.

---

## 🗂️ Domain Playbooks & Chapters
1. **[[topics/event-based-neuromorphic-vision/01-historical-evolution-and-paradigms|Historical Evolution & Paradigms]]**: From synchronous shutter frames to asynchronous logarithmic intensity events ($x, y, t, p$) and Spiking Vision Transformers (Spikformers).
2. **[[topics/event-based-neuromorphic-vision/02-production-pipeline-and-workarounds|Production Pipeline & Engineering Workarounds]]**: Event representation surfaces (Time Surfaces, Event Frames, Voxel Grids), Metavision SDK integration, and high-frequency noise filtering.
3. **[[topics/event-based-neuromorphic-vision/03-representations-and-open-problems|Representations, SNN Silicon & Open Frontiers]]**: Event-based SLAM (ESVO / EvSLAM), neuromorphic hardware processors (Intel Loihi 2, SynSense, FPGA SYNtzulu), and acoustic/vibration noise breakdown.
4. **[[topics/event-based-neuromorphic-vision/04-classical-and-hybrid-methods|Classical Event Processing & Hybrid Frame-Event Pipelines]]**: Asynchronous optical flow (Benosman et al.), classical event clustering, and hybrid RGB-Event fusion architectures.
5. **[[topics/event-based-neuromorphic-vision/README|Master Playbook & Index]]**: Curated research literature, SOTA benchmarks, commercial sensors (Prophesee GenX320), and industrial defense deployments.

---

## 🔗 Cross-Domain Connections
- Connects directly with [[topics/slam-and-spatial-perception/00-slam-and-spatial-perception-moc|SLAM MOC]] for high-speed aggressive drone odometry ($>8\text{ m/s}$).
- Interfaces with [[topics/real-time-systems/00-real-time-systems-moc|Real-Time Systems MOC]] for microsecond interrupt-driven sensor processing.
- Complements [[topics/video-tracking/00-video-tracking-moc|Video Tracking MOC]] for zero-motion-blur object tracking.**

Welcome to the **Event-Based & Neuromorphic Vision** knowledge domain. This hub covers bio-inspired asynchronous Dynamic Vision Sensors (DVS), microsecond temporal resolution, high dynamic range ($>120\text{ dB}$), Spiking Neural Networks (SNNs), and event-based SLAM.

---

## 🗂️ Domain Playbooks & Chapters
1. **[[topics/event-based-neuromorphic-vision/01-historical-evolution-and-paradigms|Historical Evolution & Paradigms]]**: From synchronous shutter frames to asynchronous logarithmic intensity events ($x, y, t, p$) and Spiking Vision Transformers (Spikformers).
2. **[[topics/event-based-neuromorphic-vision/02-production-pipeline-and-workarounds|Production Pipeline & Engineering Workarounds]]**: Event representation surfaces (Time Surfaces, Event Frames, Voxel Grids), Metavision SDK integration, and high-frequency noise filtering.
3. **[[topics/event-based-neuromorphic-vision/03-representations-and-open-problems|Representations, SNN Silicon & Open Frontiers]]**: Event-based SLAM (ESVO / EvSLAM), neuromorphic hardware processors (Intel Loihi 2, SynSense, FPGA SYNtzulu), and acoustic/vibration noise breakdown.
4. **[[topics/event-based-neuromorphic-vision/04-classical-and-hybrid-methods|Classical Event Processing & Hybrid Frame-Event Pipelines]]**: Asynchronous optical flow (Benosman et al.), classical event clustering, and hybrid RGB-Event fusion architectures.
5. **[[topics/event-based-neuromorphic-vision/README|Master Playbook & Index]]**: Curated research literature, SOTA benchmarks, commercial sensors (Prophesee GenX320), and industrial defense deployments.

---

## 🔗 Cross-Domain Connections
- Connects directly with [[topics/slam-and-spatial-perception/00-slam-and-spatial-perception-moc|SLAM MOC]] for high-speed aggressive drone odometry ($>8\text{ m/s}$).
- Interfaces with [[topics/real-time-systems/00-real-time-systems-moc|Real-Time Systems MOC]] for microsecond interrupt-driven sensor processing.
- Complements [[topics/video-tracking/00-video-tracking-moc|Video Tracking MOC]] for zero-motion-blur object tracking.

---

## 📂 All Notes in This Domain

```dataview
TABLE type AS "Note Type", updated AS "Last Updated", status AS "Status"
WHERE contains(file.folder, "topics/event-based-neuromorphic-vision")
SORT file.name ASC
```
