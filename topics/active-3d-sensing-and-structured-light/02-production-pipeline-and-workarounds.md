---
title: "Active 3D Sensing: Production Pipeline & Workarounds"
type: production-playbook
domain: Active 3D Sensing & Structured Light
tags:
  - playbook
  - engineering
  - production
  - realsense
  - tof
  - point-cloud
updated: 2026-09-08
aliases:
  - Active 3D Production Playbook
---

# 🛠️ Active 3D Sensing: Production Pipeline & Workarounds

Industrial practices for setting up RealSense D400/D455, Azure Kinect / Femto Mega, and Photoneo structured light systems.

Related notes: [[topics/active-3d-sensing-and-structured-light/00-active-3d-sensing-and-structured-light-moc|Active 3D Sensing MOC]].

---

## 1. Production Sensor Pipeline

```mermaid
flowchart LR
    Projector["Pattern Projector: IR Speckle / Fringe"] --> Sensor["IR CMOS Sensor Pair"]
    Sensor --> ASICStereo["On-Chip Vision ASIC: RealSense D4 / ToF DSP"]
    ASICStereo --> TemporalFilt["Temporal & Spatial Bilateral Depth Filter"]
    TemporalFilt --> HoleFill["Hole-Filling & Edge-Preserving Filter"]
    HoleFill --> MetricCloud["Registered Metric XYZ Point Cloud"]

```

---

## 2. Hard Real-World Engineering Gotchas & Production Fixes

### 1. The "Flying Pixels" Phenomenon at Depth Discontinuities:
- **Problem**: When a sensor pixel straddles the boundary between a foreground object ($Z_1 = 0.5\text{ m}$) and background ($Z_2 = 3.0\text{ m}$), the sensor integrates photons from both surfaces, producing an artificial intermediate depth point suspended in thin air ($Z_{\text{phantom}} \approx 1.75\text{ m}$).
- **Fix**: Apply a **Normal Vector Angle Filter** combined with a **Discontinuity Edge Erosion Filter**. Any point whose surface normal is nearly parallel ($>85^\circ$) to the camera optical axis is pruned immediately before feeding 6-DoF pose estimators.

### 2. Multi-Path Interference (MPI) in Corners & Specular Shiny Metal:
- **Problem**: In concave corners or machined aluminum parts, infrared light bounces multiple times before returning to the sensor, creating geometric distortion (concave corners appear rounded or pushed back by several centimeters).
- **Fix**: Use dual-frequency modulation ($f_1 = 20\text{ MHz}, f_2 = 80\text{ MHz}$) unwrapping or deploy active stereo (e.g. RealSense D435) rather than pure iToF when imaging metallic bins.

### 3. Laser Pattern Interference Across Multiple Cameras:
- **Problem**: Multiple structured light cameras observing the same workspace (e.g. multi-view robotic cell) blind each other due to competing projected dot patterns.
- **Fix**: Enable **Hardware External Triggering** with time-division multiplexing (TDM) so projectors pulse out-of-phase, or use active stereo with random project patterns (RealSense projectors are non-coherent and mutually tolerant).
