---
title: "Data Quality: Classical Statistics, Outlier Rejection & Hybrids"
type: production-playbook
domain: Data Quality & Verification
tags:
  - data-quality
  - chebyshev-inequality
  - isolation-forests
  - mahalanobis-distance
  - benfords-law
  - hybrid-curation
updated: 2026-09-08
aliases:
  - Data Quality Classical & Hybrid Methods
---

# 📐 Data Quality: Classical Statistics, Outlier Rejection & Hybrids

A rigorous mathematical treatment of classical statistical quality control (Chebyshev's Inequality, Grubbs' Test, Mahalanobis Distance, Benford's Law) and how classical anomaly detectors hybridize with modern foundation embedding models.

Related notes: [[topics/data-quality-and-verification/00-data-quality-and-verification-moc|Data Quality MOC]], [[topics/data-quality-and-verification/03-drift-and-open-problems|Drift & Open Frontiers]].

---

## 1. Classical Statistical Data Auditing vs. Deep Representation Auditing

```mermaid
flowchart TD
    Dataset[Raw Computer Vision Dataset: Metadata + Pixels + Annotations] --> Branch{Quality Control Methodology}
    Branch -->|Classical 1867: Chebyshev Bounds| Cheby[Distribution-Free Probability Outlier Upper Bounds]
    Branch -->|Classical 1936: Mahalanobis Metric| Maha[Covariance-Aware Distance from Semantic Cluster Centroid]
    Branch -->|Classical 1938: Benford's Law| Benford[Logarithmic First-Digit Distribution of Bounding Box Areas]
    Branch -->|Modern Hybrid 2024-2026: SOTA Paradigm| Hybrid[Foundation Model Latents -> Classical Isolation Forest / Elliptic Envelope]
    Cheby --> ProvableGuarantees[Provable Bounding Box Outlier Bounds Independent of Normality]
    Maha --> EllipsoidalOOD[Detecting Distorted Sensor Calibrations via Multivariate Covariance]
    Benford --> FraudDetection[Instant Discovery of Synthetic / Inverted Annotation Artifacts]
    Hybrid --> MicroAnomalies[Sub-Percent Defect Discovery in Massive Multimodal Datasets]
```

### Statistical Quality Verification Paradigms Compared
| Methodology | Assumptions on Distribution | Computational Complexity | Interpretability | Best Application |
| :--- | :--- | :--- | :--- | :--- |
| **Chebyshev's Inequality** | Arbitrary distribution (Finite mean/var) | $\mathcal{O}(N)$ (Single-pass mean/std) | **High (Direct mathematical bound)**| Automated pipeline bounding box dimension validation |
| **Mahalanobis Distance** | Multivariate Gaussian elliptical spread | $\mathcal{O}(D^3 + N D^2)$ (Matrix inversion)| High (Statistical covariance metric)| Sensor calibration drift & corrupted camera gain detection |
| **Isolation Forest (Liu et al.)**| Non-parametric tree partitioning | $\mathcal{O}(T \cdot \psi \log \psi)$ | Moderate | Multimodal metadata & visual vector outlier mining |
| **Hybrid (CLIP + Elliptic Envelope)**| Foundation representation geometry | High (Embedding extraction) | **High (Auditable Mahalanobis envelope)**| Zero-shot automated anomalous image pruning |

---

## 2. Mathematical Formulations: Chebyshev's Inequality & Mahalanobis Distance

### 1. Chebyshev's Inequality Bound (1867):
Unlike Gaussian $3\sigma$ rules (which falsely assume normal distributions), Chebyshev's Inequality provides a **strict mathematical upper bound on outlier probabilities for ANY arbitrary probability distribution** with finite mean $\mu$ and variance $\sigma^2$:
$$P(|X - \mu| \ge k \sigma) \le \frac{1}{k^2}$$
- Setting $k = 3 \implies P(|X - \mu| \ge 3\sigma) \le \frac{1}{9} \approx 11.1\%$
- Setting $k = 5 \implies P(|X - \mu| \ge 5\sigma) \le \frac{1}{25} = 4.0\%$
In production dataset assertion suites (e.g. Great Expectations), bounding box aspect ratios or pixel intensities exceeding $5\sigma$ can be formally flagged as statistical anomalies with provable bounds regardless of class skew.

### 2. Multivariate Mahalanobis Distance:
Given a feature vector $x \in \mathbb{R}^D$ and a dataset with mean vector $\mu \in \mathbb{R}^D$ and sample covariance matrix $\Sigma \in \mathbb{R}^{D \times D}$:
$$D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$
Unlike standard Euclidean distance $\|x - \mu\|_2$, the Mahalanobis metric:
1. Normalizes each feature by its empirical variance.
2. Decorrelates multi-channel sensor dependencies via the inverse covariance matrix $\Sigma^{-1}$, forming an elliptical contour that correctly rejects true outliers while retaining valid correlation patterns (e.g., width-to-height vehicle ratios).

---

## 3. Production Hybrid Pattern: DINOv2 Embeddings + Classical Minimum Covariance Determinant (MCD)

Deep generative models and automated web scrapers often inject subtle, corrupt images (e.g. AI-generated visual hallucinations, corrupted color balance) into training sets.

### The Production Robust Data Filter:
1. **Foundation Vectorization**: Extract 768-dimensional normalized visual embeddings $z_i$ from training images using frozen **DINOv2**.
2. **Classical FastMCD (Rousseeuw & Van Driessen)**: Fit a **Minimum Covariance Determinant (MCD)** estimator, which computes the empirical mean and covariance matrix using only the most tightly clustered $50\%$ to $75\%$ subset of the data:
   $$\hat{\mu}_{\text{MCD}}, \quad \hat{\Sigma}_{\text{MCD}}$$
   This prevents corrupt dataset poison samples from distorting the covariance estimate.
3. **Statistical Outlier Rejection**: Compute Mahalanobis distances against the robust MCD estimate. Samples exceeding the $\chi_D^2(0.999)$ chi-square critical threshold are automatically quarantined.
4. **Benefit**: Eliminates corrupted dataset poison samples without requiring human labeling.
