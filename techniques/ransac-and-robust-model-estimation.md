---
title: "RANSAC & Robust Estimation (USAC / MAGSAC++): Outlier Rejection"
type: "Technique"
domain: "Robust Estimation, Geometric Computer Vision & Feature Matching"
tags:
  - technique
  - ransac
  - robust-estimation
  - magsac
  - usac
  - prosac
  - outlier-rejection
status: evergreen
updated: 2026-09-09
aliases:
  - "RANSAC"
  - "Random Sample Consensus"
  - "MAGSAC++"
  - "USAC"
  - "PROSAC"
  - "Robust Estimation"
---

# 🛡️ RANSAC & Robust Estimation (USAC / MAGSAC++): Outlier Rejection

## 1. High-Level Concept & The Outlier Contamination Problem

In real-world computer vision, visual SLAM, and structure-from-motion (SfM):
Feature detectors and matchers (e.g., SuperPoint, SIFT, ORB, LoFTR) establish point correspondences across image pairs. However, real-world scenes feature:
- Repetitive textures (windows, bricks)
- Dynamic moving objects (pedestrians, vehicles)
- Extreme lighting changes and parallax occlusions

Consequently, correspondence sets routinely contain **$40\%\text{ to }85\%$ gross outliers** (false correspondences).

### The Breakdown Point of Least-Squares Solvers
Standard linear algebraic estimation methods (e.g., SVD, Pseudo-Inverse, Direct Linear Transform) minimize the sum of squared residuals:
$$\min_{\boldsymbol{\theta}} \sum_{i=1}^N r_i(\boldsymbol{\theta})^2$$
Because squared residuals grow quadratically ($r_i^2 \to \infty$), **a single gross outlier** situated far from the true manifold can pull the estimated model arbitrarily far off target. The breakdown point of classical least squares is **$0\%$**!

---

### The RANSAC Paradigm (Fischler & Bolles, 1981)
Instead of fitting all points simultaneously, Random Sample Consensus (RANSAC) operates via hypothesize-and-verify:
1. **Hypothesize**: Draw a random **minimal sample** of $s$ points (the minimum required to solve model parameters $\boldsymbol{\theta}$ algebraically, e.g., $s = 4$ for homography, $s = 5$ for essential matrix, $s = 3$ for plane fitting).
2. **Compute Candidate Model**: Solve the closed-form minimal solver on this sample.
3. **Verify Consensus**: Measure residuals of all $N$ data points against candidate model $\boldsymbol{\theta}$. Count inliers whose error is below threshold $\tau$ ($|r_i| \le \tau$).
4. **Iterate & Select**: Repeat for $k$ iterations and retain the model that accumulated the largest inlier support.

```
RANSAC & Modern Robust Estimator Architecture:

Data Correspondences (30% Inliers, 70% Outliers)
                 |
                 v
+--------------> [ 1. Progressive Minimal Sampling (PROSAC / Uniform) ]
|                     (Draw s points, e.g., s=4)
|                |
|                v
|                [ 2. Closed-Form Minimal Solver ]
|                     (Solve theta_hyp in microseconds)
|                |
|                v
|                [ 3. Degeneracy & Epipolar Check (DEGENSAC) ]
|                     (Reject collinear or degenerate sets)
|                |
|                v
|                [ 4. Score Consensus Support ]
|                     - Standard: Count inliers (|r| < tau)
|                     - MSAC:     Truncated quadratic loss
|                     - MAGSAC++: Marginalize across noise scales sigma
|                |
|                +---> If New Best Support: Run Local Optimization (LO-RANSAC)
|                                           Update adaptive iteration count k
|
+--- Repeat for k iterations
                 |
                 v Best Inlier Consensus Set
[ Final Non-Linear Refinement (Levenberg-Marquardt over Inliers) ]
```

---

## 2. Mathematical Formulation & Iteration Bounds

### 2.1 The Theoretical Sampling Bound
Let $w = \frac{N_{\text{inliers}}}{N_{\text{total}}}$ be the true inlier ratio in the dataset.
The probability that all $s$ randomly selected sample points are true inliers is:

$$
P(\text{all inliers in sample}) = w^s
$$

The probability that at least one point in the sample is a corrupted outlier is $1 - w^s$.
After $k$ independent random iterations, the probability that **all $k$ samples failed** (every sample contained at least one outlier) is:

$$
P(\text{all } k \text{ samples fail}) = (1 - w^s)^k
$$

We demand that the probability of succeeding at least once meets confidence $p$ (typically $p = 0.99$ or $0.999$):

$$
1 - (1 - w^s)^k \ge p \implies (1 - w^s)^k \le 1 - p
$$

Taking the natural logarithm of both sides yields the fundamental RANSAC iteration formula:

$$
k \ge \frac{\ln(1 - p)}{\ln(1 - w^s)}
$$

#### Numerical Example
For Homography estimation ($s = 4$) with confidence $p = 0.999$:
- At $w = 0.7$ (30% outliers): $w^4 = 0.2401 \implies k = \frac{\ln(0.001)}{\ln(1 - 0.2401)} \approx \mathbf{25\text{ iterations}}$.
- At $w = 0.5$ (50% outliers): $w^4 = 0.0625 \implies k \approx \mathbf{107\text{ iterations}}$.
- At $w = 0.2$ (80% outliers): $w^4 = 0.0016 \implies k \approx \mathbf{4,312\text{ iterations}}$.

---

### 2.2 Modern Enhancements: From MSAC to MAGSAC++

#### MSAC (M-Estimator SAmple Consensus)
Replaces step-function inlier counting with a continuous robust loss:
$$\rho_{\text{MSAC}}(r_i) = \begin{cases} r_i^2 & \text{if } |r_i| \le \tau \\ \tau^2 & \text{if } |r_i| > \tau \end{cases}$$
Penalizes inliers based on residual magnitude, preventing noisy hypotheses from beating accurate ones.

#### MAGSAC++ (Barath et al., 2020)
Traditional RANSAC requires manual tuning of noise threshold $\tau$. An incorrect $\tau$ causes either under-sampling or false inliers.
MAGSAC++ marginalizes over an entire range of unknown noise standard deviations $\sigma \in [0, \sigma_{\max}]$ using $\chi^2$ distributions:

$$
\text{Quality}(r_i) = \int_0^{\sigma_{\max}} P(r_i \mid \text{inlier}, \sigma) \cdot P(\sigma) \, d\sigma
$$

Yielding closed-form polynomial weighting functions that eliminate the arbitrary user threshold completely!

---

## 3. Python Reference Implementation

```python
import numpy as np

class AdaptiveRANSAC:
    """
    Adaptive RANSAC with MSAC robust scoring and dynamic sample size updating.
    """
    def __init__(self, sample_size: int = 4, threshold: float = 3.0, confidence: float = 0.999, max_iters: int = 2000):
        self.s = sample_size
        self.tau = threshold
        self.confidence = confidence
        self.max_iters = max_iters

    def fit(self, src: np.ndarray, dst: np.ndarray, solver_func, residual_func) -> tuple[np.ndarray, np.ndarray]:
        """
        src: [N, D] source points
        dst: [N, D] destination points
        solver_func: Callable(src_sample, dst_sample) -> model
        residual_func: Callable(model, src, dst) -> residuals [N]
        Returns: (best_model, inlier_mask [N])
        """
        num_points = src.shape[0]
        best_model = None
        best_score = float('inf') # MSAC minimizes loss
        best_inliers = np.zeros(num_points, dtype=bool)

        k = self.max_iters
        iteration = 0
        tau_sq = self.tau ** 2

        while iteration < k and iteration < self.max_iters:
            # 1. Uniform random minimal sample
            sample_indices = np.random.choice(num_points, self.s, replace=False)
            
            try:
                # 2. Minimal solver
                model_hyp = solver_func(src[sample_indices], dst[sample_indices])
                if model_hyp is None:
                    iteration += 1
                    continue
                
                # 3. Residual evaluation
                residuals = residual_func(model_hyp, src, dst)
                res_sq = residuals ** 2
                
                # 4. MSAC Loss: min(r^2, tau^2)
                loss = np.sum(np.where(res_sq < tau_sq, res_sq, tau_sq))
                inliers = res_sq < tau_sq
                num_inliers = np.sum(inliers)

                # 5. Check consensus update
                if loss < best_score:
                    best_score = loss
                    best_model = model_hyp
                    best_inliers = inliers

                    # Dynamically update required iterations k
                    w = max(num_inliers / num_points, 1e-6)
                    p_all_inliers = w ** self.s
                    if p_all_inliers > 0.0 and p_all_inliers < 1.0:
                        k_new = np.log(1.0 - self.confidence) / np.log(1.0 - p_all_inliers)
                        k = min(k, int(np.ceil(k_new)))

            except np.linalg.LinAlgError:
                pass

            iteration += 1

        return best_model, best_inliers
```

---

## 4. Models in the Vault Utilizing Robust Estimation

- **[[techniques/epipolar-geometry-essential-matrix|Epipolar Geometry & Essential Matrix]]**: 5-point and 8-point algorithm inlier consensus for visual odometry.
- **[[techniques/perspective-n-point-epnp-pose-estimation|Perspective-n-Point (EPnP)]]**: P3P RANSAC solving 6-DoF camera pose under 2D-to-3D visual clutter.
- **[[techniques/direct-linear-transform-and-homography|Direct Linear Transform (DLT)]]**: 4-point planar homography estimation under dynamic visual occlusion.
- **[[topics/6dof-pose-estimation/README|6-DoF Pose Estimation Playbook]]**: Outlier rejection for object pose estimation pipelines (FoundationPose).
- **[[topics/slam-and-spatial-perception/README|SLAM & Spatial Perception Playbook]]**: Loop closure correspondence verification and outlier pruning.
