---
title: "Thermal & Hyperspectral Vision MOC"
type: MOC
domain: Thermal & Hyperspectral Vision
tags:
  - moc
  - computer-vision
  - thermal-imaging
  - lwir
  - hyperspectral
  - nuc
  - sensor-fusion
status: evergreen
updated: 2026-09-08
aliases:
  - Thermal Vision MOC
  - Hyperspectral Vision Hub
---

# 🌡️ Thermal & Hyperspectral Vision MOC

> **Navigation**: [[README|🏠 Central Knowledge Hub]] / **Thermal & Hyperspectral Vision
tags:
  - moc
  - computer-vision
  - thermal-imaging
  - lwir
  - hyperspectral
  - nuc
  - sensor-fusion
status: evergreen
updated: 2026-09-08
aliases:
  - Thermal Vision MOC
  - Hyperspectral Vision Hub
---

# 🌡️ Thermal & Hyperspectral Vision MOC

Welcome to the **Thermal & Hyperspectral Vision** knowledge domain. This hub covers Long-Wave / Mid-Wave Infrared (LWIR/MWIR), uncooled microbolometers, Non-Uniformity Correction (NUC), emissivity physics, hyperspectral material identification, and frequency-guided Thermal-RGB fusion.

---

## 🗂️ Domain Playbooks & Chapters
1. **[[topics/thermal-and-hyperspectral-vision/01-historical-evolution-and-paradigms|Historical Evolution & Paradigms]]**: From Planck blackbody radiation and single-point bolometers to uncooled microbolometer FPAs, hyperspectral imaging spectrometers, and physics-informed Vision-Language Models (TherA).
2. **[[topics/thermal-and-hyperspectral-vision/02-production-pipeline-and-workarounds|Production Pipeline & Engineering Workarounds]]**: High-speed FPGA Non-Uniformity Correction (NUC), shutter calibration drift, lens thermal blooming mitigation, and cross-spectral geometric alignment.
3. **[[topics/thermal-and-hyperspectral-vision/03-sensor-physics-and-open-problems|Sensor Physics, Calibration & Open Frontiers]]**: Emissivity vs. reflectivity ambiguity, internal camera radiation noise, task-specific (UniCD) calibration, and miniaturized edge SWIR/LWIR hyperspectral classification.
4. **[[topics/thermal-and-hyperspectral-vision/04-classical-and-hybrid-methods|Classical Radiometry & Hybrid Thermal-RGB Fusion]]**: Planck**

Welcome to the **Thermal & Hyperspectral Vision** knowledge domain. This hub covers Long-Wave / Mid-Wave Infrared (LWIR/MWIR), uncooled microbolometers, Non-Uniformity Correction (NUC), emissivity physics, hyperspectral material identification, and frequency-guided Thermal-RGB fusion.

---

## 🗂️ Domain Playbooks & Chapters
1. **[[topics/thermal-and-hyperspectral-vision/01-historical-evolution-and-paradigms|Historical Evolution & Paradigms]]**: From Planck blackbody radiation and single-point bolometers to uncooled microbolometer FPAs, hyperspectral imaging spectrometers, and physics-informed Vision-Language Models (TherA).
2. **[[topics/thermal-and-hyperspectral-vision/02-production-pipeline-and-workarounds|Production Pipeline & Engineering Workarounds]]**: High-speed FPGA Non-Uniformity Correction (NUC), shutter calibration drift, lens thermal blooming mitigation, and cross-spectral geometric alignment.
3. **[[topics/thermal-and-hyperspectral-vision/03-sensor-physics-and-open-problems|Sensor Physics, Calibration & Open Frontiers]]**: Emissivity vs. reflectivity ambiguity, internal camera radiation noise, task-specific (UniCD) calibration, and miniaturized edge SWIR/LWIR hyperspectral classification.
4. **[[topics/thermal-and-hyperspectral-vision/04-classical-and-hybrid-methods|Classical Radiometry & Hybrid Thermal-RGB Fusion]]**: Planck's Law and Stefan-Boltzmann derivations, two-point polynomial NUC, and dual-backbone frequency-guided cross-attention networks.
5. **[[topics/thermal-and-hyperspectral-vision/README|Master Playbook & Index]]**: Curated research literature, public multi-modal thermal datasets (FLIR Thermal, KAIST, MODA), and industrial defense deployments.

---

## 🔗 Cross-Domain Connections
- Feeds weather-invariant inputs into [[topics/sensor-fusion/00-sensor-fusion-moc|Sensor Fusion MOC]] for zero-visibility driving through dense fog and smoke.
- Interfaces with [[topics/object-detection/00-object-detection-moc|Object Detection MOC]] for 24/7 nighttime surveillance.
- Enhances [[topics/data-quality-and-verification/00-data-quality-and-verification-moc|Data Quality MOC]] with material spectral verification.

---

## 📂 All Notes in This Domain

```dataview
TABLE type AS "Note Type", updated AS "Last Updated", status AS "Status"
WHERE contains(file.folder, "topics/thermal-and-hyperspectral-vision")
SORT file.name ASC
```
