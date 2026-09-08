---
title: "Formal Proofs & Mathematical Derivations in Computer Vision & Control"
type: didactic-derivation
domain: Mathematical Foundations & Formal Proofs
tags:
  - mathematical-proofs
  - control-barrier-functions
  - nagumo-theorem
  - liu-layland-proof
  - shapley-axioms
  - epipolar-svd
updated: 2026-09-08
aliases:
  - Formal Mathematical Derivations
  - Computer Vision Proofs Appendix
---

# 📐 Formal Proofs & Mathematical Derivations in Computer Vision & Control

A foundational mathematical reference and didactic proofs appendix establishing the exact derivations for core theorems utilized throughout the repository: **Nagumo's Theorem for Control Barrier Functions**, **The Liu & Layland Rate-Monotonic Schedulability Bound**, **The Uniqueness of the Shapley Attribution Axioms**, and **The SVD Singular Value Proof of the Essential Matrix Manifold**.

Related notes: [[topics/safety-verification-and-robustness/04-classical-and-hybrid-methods|Safety Classical Methods]], [[topics/real-time-systems/04-classical-and-hybrid-methods|Real-Time Scheduling Theory]], [[topics/explainability-and-interpretability/04-classical-and-hybrid-methods|Explainability Shapley Values]], [[topics/slam-and-spatial-perception/04-classical-and-hybrid-methods|Epipolar Geometry]].

---

## 1. Formal Proof: Nagumo's Theorem & Forward Invariance of Safe Sets

### Theorem Statement:
Consider a dynamic system $\dot{x} = f(x)$ with state space $\mathcal{X} \subseteq \mathbb{R}^n$. Let the safe set $\mathcal{C}$ be the 0-superlevel set of a continuously differentiable function $h: \mathbb{R}^n \to \mathbb{R}$:
$$\mathcal{C} = \{ x \in \mathbb{R}^n \mid h(x) \ge 0 \}, \quad \partial \mathcal{C} = \{ x \in \mathbb{R}^n \mid h(x) = 0 \}$$
The safe set $\mathcal{C}$ is **forward invariant** (if $x(0) \in \mathcal{C} \implies x(t) \in \mathcal{C} \ \forall t \ge 0$) if and only if for all boundary points $x \in \partial \mathcal{C}$:
$$\dot{h}(x) = \nabla h(x) \cdot f(x) \ge 0$$
More generally, under an extended class $\mathcal{K}_\infty$ function $\alpha(h) > 0$, the condition on the interior and boundary is:
$$\dot{h}(x) \ge -\alpha(h(x))$$

### Mathematical Derivation / Proof:
1. Let $x(t)$ be a trajectory of $\dot{x} = f(x)$ initialized at $x(0) \in \mathcal{C}$.
2. Define the scalar function $v(t) = h(x(t))$. We wish to prove that $v(t) \ge 0$ for all $t \ge 0$.
3. By the differential condition:
   $$\dot{v}(t) = \frac{d}{dt}h(x(t)) = \nabla h(x(t)) \cdot \dot{x}(t) \ge -\alpha(v(t))$$
4. Consider the linear class $\mathcal{K}$ case where $\alpha(h) = \lambda h$ with $\lambda > 0$:
   $$\dot{v}(t) + \lambda v(t) \ge 0$$
5. Multiply both sides by the positive integrating factor $e^{\lambda t}$:
   $$\frac{d}{dt}\left( e^{\lambda t} v(t) \right) = e^{\lambda t} \dot{v}(t) + \lambda e^{\lambda t} v(t) \ge 0$$
6. Integrating both sides from $0$ to $t$:
   $$\int_0^t \frac{d}{d\tau}\left( e^{\lambda \tau} v(\tau) \right) d\tau \ge 0 \implies e^{\lambda t} v(t) - v(0) \ge 0$$
7. Multiplying by $e^{-\lambda t} > 0$:
   $$v(t) \ge v(0) e^{-\lambda t}$$
8. Since $x(0) \in \mathcal{C}$, by definition $v(0) = h(x(0)) \ge 0$.
9. Because $e^{-\lambda t} > 0$ for all finite $t \ge 0$:
   $$v(t) \ge 0 \cdot e^{-\lambda t} = 0 \implies h(x(t)) \ge 0 \quad \forall t \ge 0$$
$\blacksquare$ **Conclusion**: The trajectory $x(t)$ can never cross the boundary $\partial \mathcal{C}$ into the unsafe region $h(x) < 0$. Forward invariance is mathematically proved.

---

## 2. Formal Proof: Liu & Layland Schedulability Bound for RMS

### Theorem Statement:
For $n$ independent periodic tasks scheduled under **Rate-Monotonic Scheduling (RMS)** on a single processor, the least upper bound of processor utilization ensuring guaranteed schedulability is:
$$U_{\text{lub}}(n) = n \left( 2^{1/n} - 1 \right)$$
And asymptotically:
$$\lim_{n \to \infty} U_{\text{lub}}(n) = \ln(2) \approx 0.69315 \quad (69.3\%)$$

### Mathematical Derivation / Proof:
1. Consider two periodic tasks $\tau_1 = (C_1, T_1)$ and $\tau_2 = (C_2, T_2)$ with periods $T_1 < T_2$. Under RMS, $\tau_1$ has strictly higher static priority than $\tau_2$.
2. To find the minimum utilization that can fail schedulability (the "worst-case" task set), configure $C_1, C_2$ such that $\tau_2$ just barely meets its deadline at $t = T_2$.
3. Let $q = \lfloor T_2 / T_1 \rfloor$. The most restrictive worst-case occurs when $q = 1$, meaning $1 < T_2 / T_1 < 2$.
4. During $[0, T_2]$, task $\tau_1$ executes twice (once at $t=0$ and once at $t=T_1$). For $\tau_2$ to just barely complete before $T_2$:
   $$C_2 = T_2 - 2 C_1$$
5. Compute total utilization as a function of $C_1$:
   $$U = \frac{C_1}{T_1} + \frac{C_2}{T_2} = \frac{C_1}{T_1} + \frac{T_2 - 2 C_1}{T_2} = 1 + C_1 \left( \frac{1}{T_1} - \frac{2}{T_2} \right)$$
6. Setting $C_1 = T_2 - T_1$ (the worst-case execution length where $\tau_1$ leaves minimal slack):
   $$U = \frac{T_2 - T_1}{T_1} + \frac{T_2 - 2(T_2 - T_1)}{T_2} = \frac{T_2}{T_1} - 1 + \frac{2T_1 - T_2}{T_2} = \frac{T_2}{T_1} + \frac{2 T_1}{T_2} - 1$$
7. Let ratio $R = \frac{T_2}{T_1} \in [1, 2]$. Find the minimum of $U(R) = R + \frac{2}{R} - 1$:
   $$\frac{dU}{dR} = 1 - \frac{2}{R^2} = 0 \implies R = \sqrt{2} \approx 1.414$$
8. Substitute $R = \sqrt{2}$ back into $U$:
   $$U_{\min} = \sqrt{2} + \frac{2}{\sqrt{2}} - 1 = 2\sqrt{2} - 1 = 2(2^{1/2} - 1) \approx 0.8284 \quad (82.8\%)$$
9. Extending by induction to $n$ periodic tasks, the worst-case period ratios satisfy $R_k = 2^{1/n}$, yielding the general closed-form bound:
   $$U_{\text{lub}}(n) = n \left( 2^{1/n} - 1 \right)$$
10. Evaluating the limit as $n \to \infty$ using L'Hôpital's rule:
    $$\lim_{n \to \infty} n(2^{1/n} - 1) = \lim_{x \to 0} \frac{2^x - 1}{x} = \left. \frac{d}{dx} 2^x \right|_{x=0} = \ln(2) \cdot 2^0 = \ln(2) \approx 0.69315$$
$\blacksquare$ **Conclusion**: Any real-time task set with total processor utilization $U \le 69.3\%$ is guaranteed to never miss a hard deadline under RMS.

---

## 3. Formal Proof: Uniqueness of Lloyd Shapley's Attribution Formula

### Theorem Statement:
Let $N = \{1, \dots, M\}$ be the set of all input features, and let $v(S)$ be the characteristic function defining the model output on feature subset $S \subseteq N$ (with $v(\emptyset) = 0$).
There is **one and only one** attribution mapping $\phi: v \to \mathbb{R}^M$ that simultaneously satisfies four fundamental axioms:
1. **Efficiency**: $\sum_{i \in N} \phi_i(v) = v(N)$
2. **Symmetry**: If $v(S \cup \{i\}) = v(S \cup \{j\}) \ \forall S \subseteq N \setminus \{i, j\}$, then $\phi_i(v) = \phi_j(v)$
3. **Linearity**: For two models $u, v$ and constants $a, b$: $\phi(a u + b v) = a \phi(u) + b \phi(v)$
4. **Null Player (Dummy)**: If $v(S \cup \{i\}) = v(S) \ \forall S \subseteq N \setminus \{i\}$, then $\phi_i(v) = 0$

### Mathematical Derivation / Proof (Sketch):
1. Because the mapping $v \mapsto \phi(v)$ is linear (Axiom 3), it suffices to determine $\phi$ on a vector space basis of game functions.
2. For any non-empty coalition $R \subseteq N$, define the **unanimity game** $u_R$:
   $$u_R(S) = \begin{cases} 1 & \text{if } R \subseteq S \\ 0 & \text{otherwise} \end{cases}$$
   The set of all $2^M - 1$ unanimity games $\{ u_R \mid \emptyset \ne R \subseteq N \}$ forms a complete linear basis for all games $v$.
3. For a unanimity game $u_R$:
   - Any player $i \notin R$ is a **Null Player** because adding $i$ to $S$ never changes whether $R \subseteq S$. By Axiom 4: $\phi_i(u_R) = 0 \ \forall i \notin R$.
   - All players $i, j \in R$ are completely **Symmetric**. By Axiom 2: $\phi_i(u_R) = \phi_j(u_R) = c$ for all $i \in R$.
   - By **Efficiency** (Axiom 1):
     $$\sum_{i \in N} \phi_i(u_R) = \sum_{i \in R} \phi_i(u_R) = |R| \cdot c = u_R(N) = 1 \implies c = \frac{1}{|R|}$$
   - Therefore, on every basis game $u_R$, the attribution is uniquely forced to be:
     $$\phi_i(u_R) = \begin{cases} \frac{1}{|R|} & \text{if } i \in R \\ 0 & \text{if } i \notin R \end{cases}$$
4. Every arbitrary game $v$ has a unique linear expansion $v = \sum_{R \subseteq N} c_R u_R$.
5. Applying linearity, $\phi(v)$ is uniquely determined on all games, yielding the famous combinatorial probability formula:
   $$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! (|N| - |S| - 1)!}{|N|!} \left[ v(S \cup \{i\}) - v(S) \right]$$
$\blacksquare$ **Conclusion**: The Shapley Value is the mathematically unique solution for game-theoretic feature attribution.

---

## 4. Formal Proof: Essential Matrix Singular Value Manifold Constraints

### Theorem Statement (Huang & Faugeras, 1989):
A real $3 \times 3$ matrix $E$ is an **Essential Matrix** corresponding to a rigid displacement $E = [t]_\times R$ (where $R \in SO(3)$ and $t \in \mathbb{R}^3, t \ne 0$) if and only if its Singular Value Decomposition (SVD) satisfies:
$$E = U \text{diag}(\sigma, \sigma, 0) V^T \quad \text{for some } \sigma > 0, \ U, V \in SO(3)$$
That is: **the two non-zero singular values of $E$ must be strictly equal, and the third singular value must be zero.**

### Mathematical Derivation / Proof:
1. Express $E$ as $E = [t]_\times R$.
2. Compute the symmetric matrix $E E^T$:
   $$E E^T = ([t]_\times R) ([t]_\times R)^T = [t]_\times R R^T [t]_\times^T$$
3. Since $R \in SO(3)$, $R R^T = I$:
   $$E E^T = [t]_\times [t]_\times^T$$
4. Since $[t]_\times$ is skew-symmetric, $[t]_\times^T = -[t]_\times$:
   $$E E^T = - [t]_\times^2$$
5. Using Lagrange's vector cross-product identity: $[t]_\times^2 = t t^T - \|t\|_2^2 I$:
   $$E E^T = - (t t^T - \|t\|_2^2 I) = \|t\|_2^2 I - t t^T$$
6. Determine the eigenvalues of $E E^T$:
   - For eigenvector $v_1 = t$:
     $$(E E^T) t = (\|t\|_2^2 I - t t^T) t = \|t\|_2^2 t - t (t^T t) = \|t\|_2^2 t - \|t\|_2^2 t = 0$$
     Thus, $\lambda_3 = 0$ with eigenvector parallel to $t$.
   - For any vector $v_\perp$ orthogonal to $t$ ($t^T v_\perp = 0$):
     $$(E E^T) v_\perp = (\|t\|_2^2 I - t t^T) v_\perp = \|t\|_2^2 v_\perp - 0 = \|t\|_2^2 v_\perp$$
     Thus, the orthogonal 2D subspace yields two identical eigenvalues: $\lambda_1 = \lambda_2 = \|t\|_2^2$.
7. The singular values of $E$ are the square roots of the eigenvalues of $E E^T$:
   $$\sigma_1 = \sqrt{\lambda_1} = \|t\|_2, \quad \sigma_2 = \sqrt{\lambda_2} = \|t\|_2, \quad \sigma_3 = \sqrt{\lambda_3} = 0$$
$\blacksquare$ **Conclusion**: Any geometrically valid essential matrix must have two identical non-zero singular values ($\sigma_1 = \sigma_2 = \|t\|$) and one zero singular value ($\sigma_3 = 0$). Projecting arbitrary estimated matrices onto this manifold guarantees physical rigidity.
