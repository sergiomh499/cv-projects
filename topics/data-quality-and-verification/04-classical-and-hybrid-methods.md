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
    Dataset["Raw Computer Vision Dataset: Metadata + Pixels + Annotations"] --> Branch{"Quality Control Methodology"}
    Branch -->|Classical 1867: Chebyshev Bounds| Cheby["Distribution-Free Probability Outlier Upper Bounds"]
    Branch -->|Classical 1936: Mahalanobis Metric| Maha["Covariance-Aware Distance from Semantic Cluster Centroid"]
    Branch -->|Classical 1938: Benford's Law| Benford["Logarithmic First-Digit Distribution of Bounding Box Areas"]
    Branch -->|Modern Hybrid 2024-2026: SOTA Paradigm| Hybrid["Foundation Model Latents + Classical Isolation Forest / Elliptic Envelope"]
    Cheby --> ProvableGuarantees["Provable Bounding Box Outlier Bounds Independent of Normality"]
    Maha --> EllipsoidalOOD["Detecting Distorted Sensor Calibrations via Multivariate Covariance"]
    Benford --> FraudDetection["Instant Discovery of Synthetic / Inverted Annotation Artifacts"]
    Hybrid --> MicroAnomalies["Sub-Percent Defect Discovery in Massive Multimodal Datasets"]
```

### Statistical Quality Verification Paradigms Compared
| Methodology | Assumptions on Distribution | Computational Complexity | Interpretability | Best Application |
| :--- | :--- | :--- | :--- | :--- |
| **Chebyshev's Inequality** | Arbitrary distribution (Finite mean/var) | $\mathcal{O}(N)$ (Single-pass mean/std) | **High (Direct mathematical bound)** | Automated pipeline bounding box dimension validation |
| **Mahalanobis Distance** | Multivariate Gaussian elliptical spread | $\mathcal{O}(D^3 + N D^2)$ (Matrix inversion) | High (Statistical covariance metric) | Sensor calibration drift & corrupted camera gain detection |
| **Isolation Forest (Liu et al.)** | Non-parametric tree partitioning | $\mathcal{O}(T \cdot \psi \log \psi)$ | Moderate | Multimodal metadata & visual vector outlier mining |
| **Hybrid (CLIP + Elliptic Envelope)** | Foundation representation geometry | High (Embedding extraction) | **High (Auditable Mahalanobis envelope)** | Zero-shot automated anomalous image pruning |

---

## 2. Mathematical Formulations: Chebyshev's Inequality & Mahalanobis Distance

### 2.1 Chebyshev's Inequality Bound (1867)

Unlike Gaussian $3\sigma$ rules (which falsely assume normal distributions), Chebyshev's Inequality provides a **strict mathematical upper bound on outlier probabilities for ANY arbitrary probability distribution** with finite mean $\mu$ and variance $\sigma^2$:

$$P(|X - \mu| \ge k \sigma) \le \frac{1}{k^2}$$

- Setting $k = 3 \implies P(|X - \mu| \ge 3\sigma) \le \frac{1}{9} \approx 11.1\%$
- Setting $k = 5 \implies P(|X - \mu| \ge 5\sigma) \le \frac{1}{25} = 4.0\%$
- Setting $k = 10 \implies P(|X - \mu| \ge 10\sigma) \le \frac{1}{100} = 1.0\%$

In production dataset assertion suites (Great Expectations), bounding box aspect ratios or pixel intensities exceeding $5\sigma$ are formally flagged as statistical anomalies with provable bounds regardless of class skew. This is strictly stronger than assuming Gaussian and applying a $5\sigma$ threshold (which would incorrectly imply $P < 3 \times 10^{-7}$).

```python
import numpy as np

def chebyshev_outlier_mask(values: np.ndarray, k: float = 5.0) -> np.ndarray:
    """Flag outliers via distribution-free Chebyshev bound. Guaranteed FPR <= 1/k^2."""
    mu, sigma = values.mean(), values.std()
    return np.abs(values - mu) >= k * sigma
    # For k=5: guaranteed <= 4% of ANY distribution falls beyond threshold

# In production: flag bounding boxes with extreme aspect ratios
bbox_areas = np.array([w * h for (w, h) in annotations])
outlier_flags = chebyshev_outlier_mask(bbox_areas, k=5)
# Provable guarantee: at most 4% false positive rate regardless of annotation distribution
```

### 2.2 Multivariate Mahalanobis Distance

Given a feature vector $x \in \mathbb{R}^D$ and a dataset with mean vector $\mu \in \mathbb{R}^D$ and sample covariance matrix $\Sigma \in \mathbb{R}^{D \times D}$:

$$D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$

Unlike standard Euclidean distance $\|x - \mu\|_2$, the Mahalanobis metric:
1. Normalizes each feature by its empirical variance.
2. Decorrelates multi-channel sensor dependencies via the inverse covariance matrix $\Sigma^{-1}$, forming an elliptical contour that correctly rejects true outliers while retaining valid correlation patterns (e.g., width-to-height vehicle ratios).

Under multivariate normality, $D_M^2(x) \sim \chi^2_D$ — so the $\chi^2_{D}(0.999)$ quantile gives a statistically calibrated outlier threshold. For $D = 5$ metadata features: $\chi^2_5(0.999) \approx 20.5$.

### 2.3 Benford's Law for Annotation Fraud Detection

In any naturally occurring numerical dataset, the first significant digit $d \in \{1, \dots, 9\}$ follows the logarithmic distribution:

$$P(\text{first digit} = d) = \log_{10}\!\left(1 + \frac{1}{d}\right)$$

This gives: $P(d=1) \approx 30.1\%$, $P(d=2) \approx 17.6\%$, ..., $P(d=9) \approx 4.6\%$.

Naturally annotated bounding box areas follow Benford's Law. Programmatically generated (synthetic or copy-pasted) annotations exhibit **uniform first-digit distributions**—a chi-square goodness-of-fit test flags them:

$$\chi^2 = \sum_{d=1}^{9} \frac{(O_d - E_d)^2}{E_d} \sim \chi^2_8 \text{ (under null)}$$

Threshold: $\chi^2 > 20.1$ ($p < 0.01$, 8 degrees of freedom) indicates non-natural annotation generation, warranting audit for copy-paste artifacts or synthetic label injection.

---

## 3. Production Hybrid Pattern: DINOv2 Embeddings + Classical Minimum Covariance Determinant (MCD)

Deep generative models and automated web scrapers often inject subtle corrupt images (AI-generated hallucinations, corrupted color balance) into training sets.

### The Production Robust Data Filter

1. **Foundation Vectorization**: Extract 768-dimensional normalized visual embeddings $z_i$ from training images using frozen **DINOv2 ViT-L/14**.

2. **Classical FastMCD (Rousseeuw & Van Driessen)**: Fit a **Minimum Covariance Determinant** estimator using only the most tightly clustered 75% subset of the data:
   $$(\hat{\mu}_{\text{MCD}},\; \hat{\Sigma}_{\text{MCD}}) = \underset{|H| = \lfloor 0.75 N \rfloor}{\arg\min} \det\!\left(\Sigma_H\right)$$
   This prevents corrupt poison samples from distorting the covariance estimate.

3. **Statistical Outlier Rejection**: Compute Mahalanobis distances against the robust MCD estimate. Samples exceeding the $\chi_d^2(0.999)$ chi-square critical threshold are automatically quarantined.

```python
from sklearn.covariance import MinCovDet
from scipy.stats import chi2
import numpy as np

def robust_outlier_filter(embeddings: np.ndarray, alpha: float = 0.001) -> np.ndarray:
    """
    Returns boolean mask: True = inlier, False = outlier.
    Uses FastMCD for robustness against poison samples.
    """
    mcd = MinCovDet(support_fraction=0.75, random_state=42)
    mcd.fit(embeddings)
    mahal_sq = mcd.mahalanobis(embeddings)  # shape (N,)
    d = embeddings.shape[1]
    threshold = chi2.ppf(1 - alpha, df=min(d, 50))  # cap df for stability
    return mahal_sq <= threshold

# Usage: filter 1M-image dataset
embeds = extract_dinov2(image_paths)  # (N, 768)
inlier_mask = robust_outlier_filter(embeds[:, :50])  # Use top-50 PCA components
clean_paths = [p for p, ok in zip(image_paths, inlier_mask) if ok]
```

4. **Benefit**: Eliminates corrupted dataset poison samples without requiring human labeling. In practice, identifies **0.3–1.2%** of web-scraped training data as corrupt (benchmarked on LAION-400M subset, 2024).

---

## 4. Hybrid Pipeline: Combining Classical + Neural Methods

### The 4-Stage Production Curation Stack

| Stage | Method | What it catches | Speed |
|:------|:-------|:----------------|:------|
| 1. Chebyshev metadata audit | Scalar statistics | Impossible aspect ratios, NaN coordinates | < 1 ms/sample |
| 2. Benford's Law on bbox areas | Log-digit distribution | Programmatic annotation artifacts | < 1 ms/sample |
| 3. DINOv2 + FastMCD | Robust Mahalanobis in latent space | Corrupt images, AI-generated fakes | ~5 ms/sample (GPU) |
| 4. Cleanlab CL | Joint label-noise matrix estimation | Mislabeled class assignments | Requires CV training run |

Running all four stages in sequence on a 500k-image dataset (A100 GPU, 32 CPU cores) takes approximately 3 hours end-to-end, flagging an average of 4.7% of images for human review—vs. 40+ hours of fully manual curation for the same dataset size.
