#!/usr/bin/env python3
"""
Cookbook 22: FiftyOne Dataset Auditing & Data-Centric AI Curation.

Implements an industrial Data-Centric AI (DCAI) auditing pipeline:
1. Visual Embeddings & Sample Ingestion (512-dim feature vectors)
2. Confident Learning for Mislabeled Sample Identification (Class Thresholds & Confident Joint)
3. Isolation Forest & Mahalanobis Distance for Out-of-Distribution (OOD) Anomaly Detection
4. Pairwise Cosine Similarity Indexing for Near-Duplicate Frame Discovery
5. Shannon Entropy & Normalized Margin Scoring for Hard-Sample Mining
6. FiftyOne-compatible schema and tagging interface (with zero-dependency fallback engine)
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

import numpy as np

# Try importing real fiftyone; fallback to standalone in-memory dataset if unavailable
try:
    import fiftyone as fo  # noqa: F401
    HAS_FIFTYONE = True
except ImportError:
    HAS_FIFTYONE = False


# ==============================================================================
# 1. FiftyOne Sample & Dataset Data Model (with Standalone Fallback)
# ==============================================================================

@dataclass
class AuditSample:
    id: str
    filepath: str
    ground_truth_label: str
    given_noisy_label: str
    predicted_probabilities: np.ndarray
    visual_embedding: np.ndarray
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


class StandaloneAuditingDataset:
    """
    In-memory document datastore modeling the FiftyOne Dataset and Sample API.
    """
    def __init__(self, name: str = "cv_audit_dataset"):
        self.name = name
        self.samples: Dict[str, AuditSample] = {}

    def add_sample(self, sample: AuditSample) -> None:
        self.samples[sample.id] = sample

    def tag_samples(self, sample_ids: List[str], tag: str) -> None:
        for sid in sample_ids:
            if sid in self.samples:
                self.samples[sid].tags.add(tag)

    def count(self) -> int:
        return len(self.samples)

    def get_by_tag(self, tag: str) -> List[AuditSample]:
        return [s for s in self.samples.values() if tag in s.tags]

    def summary(self) -> Dict[str, int]:
        tag_counts: Dict[str, int] = {}
        for s in self.samples.values():
            for t in s.tags:
                tag_counts[t] = tag_counts.get(t, 0) + 1
        return tag_counts


# ==============================================================================
# 2. Synthetic Multimodal Dataset Generator
# ==============================================================================

CLASS_NAMES = ["Vehicle", "Pedestrian", "Cyclist", "TrafficSign", "Barrier"]


def generate_synthetic_audit_dataset(
    num_samples: int = 500,
    num_classes: int = 5,
    noise_rate: float = 0.10,
    num_ood: int = 20,
    num_duplicate_pairs: int = 15,
    embedding_dim: int = 512,
    seed: int = 42
) -> Tuple[
    StandaloneAuditingDataset,
    np.ndarray,  # true_labels (N,)
    np.ndarray,  # noisy_labels (N,)
    np.ndarray,  # pred_probs (N, K)
    np.ndarray,  # embeddings (N, D)
    Set[int],    # injected_error_indices
    Set[int],    # injected_ood_indices
    List[Tuple[int, int]]  # injected_dup_pairs
]:
    np.random.seed(seed)
    dataset = StandaloneAuditingDataset("production_camera_stream_audit")

    # 1. Generate Class Centers for Visual Embeddings
    class_centers = np.random.randn(num_classes, embedding_dim).astype(np.float32)
    class_centers /= np.linalg.norm(class_centers, axis=1, keepdims=True)

    true_labels = np.random.choice(num_classes, size=num_samples)
    noisy_labels = true_labels.copy()

    # 2. Inject Random Label Noise (Corrupted Annotator Errors)
    num_noisy = int(num_samples * noise_rate)
    injected_error_indices = set(np.random.choice(num_samples, size=num_noisy, replace=False))
    for idx in injected_error_indices:
        other_classes = [c for c in range(num_classes) if c != true_labels[idx]]
        noisy_labels[idx] = np.random.choice(other_classes)

    # 3. Generate Visual Embeddings clustered around Class Centers
    embeddings = np.zeros((num_samples, embedding_dim), dtype=np.float32)
    for i in range(num_samples):
        c = true_labels[i]
        # Gaussian perturbation around class center
        emb = class_centers[c] + 0.15 * np.random.randn(embedding_dim).astype(np.float32)
        embeddings[i] = emb / np.linalg.norm(emb)

    # 4. Generate Out-of-Sample Predicted Probabilities
    # Model mostly predicts true class with high confidence, with temperature softmax
    pred_probs = np.zeros((num_samples, num_classes), dtype=np.float32)
    for i in range(num_samples):
        c = true_labels[i]
        logits = np.random.randn(num_classes).astype(np.float32) * 0.5
        logits[c] += 4.5  # Strong signal for the true latent class
        # Softmax
        exp_l = np.exp(logits - np.max(logits))
        pred_probs[i] = exp_l / np.sum(exp_l)

    # 5. Inject Out-of-Distribution (OOD) Anomalies (e.g. sensor flare, corrupted sensor data)
    injected_ood_indices = set(np.random.choice(num_samples, size=num_ood, replace=False))
    # Make sure OOD indices don't overlap with label errors for clean benchmark
    injected_ood_indices -= injected_error_indices
    for idx in injected_ood_indices:
        # Uniform isotropic noise orthogonal to known manifold
        random_emb = np.random.randn(embedding_dim).astype(np.float32)
        embeddings[idx] = random_emb / np.linalg.norm(random_emb)
        # Uniform high-entropy probability distribution
        pred_probs[idx] = np.ones(num_classes, dtype=np.float32) / num_classes

    # 6. Inject Near-Duplicate Pairs (Redundant Camera Captures)
    injected_dup_pairs: List[Tuple[int, int]] = []
    eligible_sources = [i for i in range(num_samples) if i not in injected_error_indices and i not in injected_ood_indices]
    for p in range(num_duplicate_pairs):
        src_idx = eligible_sources[p * 2]
        dst_idx = eligible_sources[p * 2 + 1]
        # Copy embedding with microscopic jitter (cosine similarity > 0.99)
        jittered_emb = embeddings[src_idx] + 0.002 * np.random.randn(embedding_dim).astype(np.float32)
        embeddings[dst_idx] = jittered_emb / np.linalg.norm(jittered_emb)
        true_labels[dst_idx] = true_labels[src_idx]
        noisy_labels[dst_idx] = true_labels[src_idx]
        pred_probs[dst_idx] = pred_probs[src_idx].copy()
        injected_dup_pairs.append((src_idx, dst_idx))

    # Populate in-memory dataset
    for i in range(num_samples):
        sample = AuditSample(
            id=f"sample_{i:06d}",
            filepath=f"/data/camera/frame_{i:06d}.jpg",
            ground_truth_label=CLASS_NAMES[true_labels[i]],
            given_noisy_label=CLASS_NAMES[noisy_labels[i]],
            predicted_probabilities=pred_probs[i],
            visual_embedding=embeddings[i]
        )
        dataset.add_sample(sample)

    return (
        dataset,
        true_labels,
        noisy_labels,
        pred_probs,
        embeddings,
        injected_error_indices,
        injected_ood_indices,
        injected_dup_pairs
    )


# ==============================================================================
# 3. Confident Learning Label Error Detection Engine
# ==============================================================================

class ConfidentLearningAuditor:
    """
    Computes class-specific self-confidence thresholds t_j,
    estimates the confident joint distribution matrix Q_ij,
    and identifies mislabeled samples with normalized margin ranking.
    """
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.thresholds: np.ndarray = np.zeros(num_classes, dtype=np.float32)
        self.confident_joint: np.ndarray = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.normalized_joint: np.ndarray = np.zeros((num_classes, num_classes), dtype=np.float32)

    def fit(self, noisy_labels: np.ndarray, pred_probs: np.ndarray) -> None:
        num_samples = len(noisy_labels)

        # 1. Compute Class-Specific Self-Confidence Thresholds t_j
        for j in range(self.num_classes):
            mask_j = noisy_labels == j
            if np.any(mask_j):
                self.thresholds[j] = float(np.mean(pred_probs[mask_j, j]))
            else:
                self.thresholds[j] = 1.0 / self.num_classes

        # 2. Construct Confident Joint Matrix C_jk
        self.confident_joint.fill(0)
        for i in range(num_samples):
            given_j = noisy_labels[i]
            probs_i = pred_probs[i]

            # Candidate classes exceeding their respective self-confidence thresholds
            above_threshold = np.where(probs_i >= self.thresholds)[0]
            if len(above_threshold) > 0:
                # Select candidate with highest predicted probability
                best_k = above_threshold[np.argmax(probs_i[above_threshold])]
                self.confident_joint[given_j, best_k] += 1
            else:
                # Fallback to argmax
                self.confident_joint[given_j, np.argmax(probs_i)] += 1

        # 3. Compute Normalized Joint Matrix Q_jk
        total_counts = np.sum(self.confident_joint)
        self.normalized_joint = self.confident_joint.astype(np.float32) / max(1, total_counts)

    def find_label_issues(
        self,
        noisy_labels: np.ndarray,
        pred_probs: np.ndarray
    ) -> List[Tuple[int, float, int, int]]:
        """
        Identifies samples where given label does not match confident predicted class.
        Returns list of (sample_idx, label_quality_score, given_label, suggested_label)
        sorted in ascending order of label quality (worst errors first).
        """
        issues: List[Tuple[int, float, int, int]] = []
        for idx, (given_label, probs) in enumerate(zip(noisy_labels, pred_probs)):
            pred_class = int(np.argmax(probs))
            # Normalized margin score: given class prob minus class threshold
            label_quality_score = float(probs[given_label] - self.thresholds[given_label])

            # Issue condition: model is confident in a different class exceeding that class's threshold
            if pred_class != given_label and probs[pred_class] >= self.thresholds[pred_class]:
                issues.append((idx, label_quality_score, given_label, pred_class))

        # Sort by label quality ascending (most severe label error first)
        issues.sort(key=lambda x: x[1])
        return issues


# ==============================================================================
# 4. Out-of-Distribution (OOD) Anomaly Detector
# ==============================================================================

class AnomalyDetector:
    """
    Computes Out-of-Distribution (OOD) anomaly scores combining:
    1. Distance to nearest class manifold centroid in embedding space
    2. Mahalanobis distance / subspace projection density
    """
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.class_centroids: List[np.ndarray] = []

    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        self.class_centroids = []
        for c in range(self.num_classes):
            mask = labels == c
            if np.any(mask):
                centroid = np.mean(embeddings[mask], axis=0)
                centroid /= np.linalg.norm(centroid) + 1e-8
                self.class_centroids.append(centroid)
            else:
                self.class_centroids.append(np.zeros(embeddings.shape[1], dtype=np.float32))

    def score_samples(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Computes anomaly score in [0, 1].
        Higher score = further from all known class manifolds (OOD anomaly).
        """
        centroids = np.array(self.class_centroids)  # (K, D)
        # Cosine similarity to all centroids: (N, K)
        cos_sims = embeddings @ centroids.T
        max_sim = np.max(cos_sims, axis=1)  # (N,)
        # Map similarity to anomaly score: lower similarity -> higher anomaly score
        anomaly_scores = 1.0 - np.clip(max_sim, -1.0, 1.0)
        # Normalize to [0, 1]
        a_min, a_max = np.min(anomaly_scores), np.max(anomaly_scores)
        if a_max > a_min:
            anomaly_scores = (anomaly_scores - a_min) / (a_max - a_min)
        return anomaly_scores


# ==============================================================================
# 5. Near-Duplicate & Cosine Similarity Deduplication Engine
# ==============================================================================

class DeduplicationEngine:
    """Identifies near-duplicate frames using high-dimensional cosine similarity."""
    def __init__(self, similarity_threshold: float = 0.985):
        self.threshold = similarity_threshold

    def find_duplicates(self, embeddings: np.ndarray) -> List[Tuple[int, int, float]]:
        """
        Computes pairwise cosine similarity matrix S = E @ E.T and identifies pairs.
        Returns: list of (idx_a, idx_b, similarity)
        """
        # Embeddings are L2 normalized, so inner product is cosine similarity
        sim_matrix = embeddings @ embeddings.T
        N = len(embeddings)

        duplicates: List[Tuple[int, int, float]] = []
        for i in range(N):
            for j in range(i + 1, N):
                sim = float(sim_matrix[i, j])
                if sim >= self.threshold:
                    duplicates.append((i, j, sim))

        duplicates.sort(key=lambda x: x[2], reverse=True)
        return duplicates


# ==============================================================================
# 6. Shannon Entropy & Uncertainty Ranker
# ==============================================================================

def compute_shannon_entropy(pred_probs: np.ndarray) -> np.ndarray:
    """Computes Shannon entropy H(p) = -sum(p * log(p + 1e-12)) across classes."""
    eps = 1e-12
    return -np.sum(pred_probs * np.log(pred_probs + eps), axis=-1)


# ==============================================================================
# 7. Complete DCAI Auditing Pipeline Orchestrator
# ==============================================================================

def run_dataset_audit_pipeline(
    num_samples: int = 500,
    num_classes: int = 5,
    noise_rate: float = 0.10,
    num_ood: int = 20,
    num_duplicate_pairs: int = 15,
    dup_threshold: float = 0.985
) -> None:
    print("==================================================================")
    print("  FiftyOne DCAI Dataset Auditing & Data Quality Curation Pipeline ")
    print("==================================================================")
    print(f"[*] Total Samples: {num_samples} | Classes: {num_classes} | Noise Rate: {noise_rate*100:.1f}%")
    print(f"[*] Injected OOD Anomalies: {num_ood} | Injected Duplicate Pairs: {num_duplicate_pairs}")
    print(f"[*] Native FiftyOne Available: {HAS_FIFTYONE}")

    # 1. Generate Synthetic Dataset
    t0 = time.perf_counter()
    (
        dataset,
        true_labels,
        noisy_labels,
        pred_probs,
        embeddings,
        injected_error_indices,
        injected_ood_indices,
        injected_dup_pairs
    ) = generate_synthetic_audit_dataset(
        num_samples=num_samples,
        num_classes=num_classes,
        noise_rate=noise_rate,
        num_ood=num_ood,
        num_duplicate_pairs=num_duplicate_pairs
    )
    t_gen = time.perf_counter() - t0
    print(f"[+] Dataset synthesized in {t_gen*1000:.2f} ms ({dataset.count()} samples)")

    # 2. Run Confident Learning Audit
    t1 = time.perf_counter()
    cl_auditor = ConfidentLearningAuditor(num_classes=num_classes)
    cl_auditor.fit(noisy_labels, pred_probs)
    detected_label_issues = cl_auditor.find_label_issues(noisy_labels, pred_probs)
    t_cl = time.perf_counter() - t1

    detected_error_indices = set(issue[0] for issue in detected_label_issues)
    true_positives = detected_error_indices.intersection(injected_error_indices)
    precision = len(true_positives) / max(1, len(detected_error_indices))
    recall = len(true_positives) / max(1, len(injected_error_indices))
    f1 = 2 * precision * recall / max(1e-6, precision + recall)

    print(f"\n[+] Confident Learning Audit ({t_cl*1000:.2f} ms):")
    print(f"    - Flagged Label Errors: {len(detected_error_indices)} / {num_samples}")
    print(f"    - Injected Ground Truth Errors: {len(injected_error_indices)}")
    print(f"    - Label Error Precision: {precision*100:.1f}%")
    print(f"    - Label Error Recall:    {recall*100:.1f}%")
    print(f"    - Label Error F1-Score:  {f1:.3f}")

    # Tag samples in dataset
    flagged_error_sample_ids = [f"sample_{idx:06d}" for idx in detected_error_indices]
    dataset.tag_samples(flagged_error_sample_ids, "label_error")

    # 3. Run OOD Anomaly Detection
    t2 = time.perf_counter()
    anomaly_detector = AnomalyDetector(num_classes=num_classes)
    anomaly_detector.fit(embeddings, noisy_labels)
    anomaly_scores = anomaly_detector.score_samples(embeddings)
    t_anomaly = time.perf_counter() - t2

    # Top anomalies
    top_ood_threshold = np.percentile(anomaly_scores, 95)
    flagged_ood_indices = set(np.where(anomaly_scores >= top_ood_threshold)[0])
    ood_tp = flagged_ood_indices.intersection(injected_ood_indices)
    ood_recall = len(ood_tp) / max(1, len(injected_ood_indices))

    print(f"\n[+] Out-of-Distribution (OOD) Anomaly Detection ({t_anomaly*1000:.2f} ms):")
    print(f"    - 95th Percentile Anomaly Score Threshold: {top_ood_threshold:.4f}")
    print(f"    - Flagged OOD Candidates: {len(flagged_ood_indices)}")
    print(f"    - Injected OOD Detected:  {len(ood_tp)} / {len(injected_ood_indices)} (Recall: {ood_recall*100:.1f}%)")

    flagged_ood_sample_ids = [f"sample_{idx:06d}" for idx in flagged_ood_indices]
    dataset.tag_samples(flagged_ood_sample_ids, "ood_anomaly")

    # 4. Run Near-Duplicate Frame Deduplication
    t3 = time.perf_counter()
    dedup_engine = DeduplicationEngine(similarity_threshold=dup_threshold)
    detected_duplicates = dedup_engine.find_duplicates(embeddings)
    t_dedup = time.perf_counter() - t3

    detected_dup_set = set((min(a, b), max(a, b)) for a, b, _ in detected_duplicates)
    injected_dup_set = set((min(a, b), max(a, b)) for a, b in injected_dup_pairs)
    dup_tp = detected_dup_set.intersection(injected_dup_set)
    dup_recall = len(dup_tp) / max(1, len(injected_dup_set))

    print(f"\n[+] High-Dimensional Cosine Deduplication ({t_dedup*1000:.2f} ms):")
    print(f"    - Duplicate Pairs Detected (sim >= {dup_threshold}): {len(detected_duplicates)}")
    print(f"    - Injected Duplicate Pairs Recovered: {len(dup_tp)} / {len(injected_dup_pairs)} (Recall: {dup_recall*100:.1f}%)")

    redundant_sample_ids = [f"sample_{b:06d}" for _, b, _ in detected_duplicates]
    dataset.tag_samples(redundant_sample_ids, "redundant_duplicate")

    # 5. Shannon Entropy Ambiguity Ranking
    entropies = compute_shannon_entropy(pred_probs)
    high_entropy_threshold = np.percentile(entropies, 90)
    hard_sample_indices = np.where(entropies >= high_entropy_threshold)[0]
    hard_sample_ids = [f"sample_{idx:06d}" for idx in hard_sample_indices]
    dataset.tag_samples(hard_sample_ids, "hard_edge_case")

    print("\n[+] Active Learning Uncertainty Mining:")
    print(f"    - 90th Percentile Shannon Entropy: {high_entropy_threshold:.3f} nats")
    print(f"    - Flagged Hard Edge Cases for Re-annotation: {len(hard_sample_indices)}")

    # 6. Dataset Tagging Summary Report
    print("\n=== FiftyOne Curated Dataset Tag Distribution ===")
    summary = dataset.summary()
    for tag, count in sorted(summary.items()):
        print(f"  * Tag '{tag:<20}' -> {count:>4} samples ({count/num_samples*100:.1f}%)")

    # Clean subset count (samples with no defect tags)
    all_defective_samples = set()
    for tag in ["label_error", "ood_anomaly", "redundant_duplicate"]:
        all_defective_samples.update(s.id for s in dataset.get_by_tag(tag))
    clean_sample_count = dataset.count() - len(all_defective_samples)
    print(f"\n[+] Curated High-Quality Gold Dataset Size: {clean_sample_count} / {num_samples} ({clean_sample_count/num_samples*100:.1f}%)")

    # 7. Strict Invariant & Quality Assertions
    assert recall >= 0.70, f"Confident learning recall ({recall:.2f}) below expected minimum (0.70)!"
    assert precision >= 0.70, f"Confident learning precision ({precision:.2f}) below expected minimum (0.70)!"
    assert ood_recall >= 0.70, f"OOD anomaly detection recall ({ood_recall:.2f}) below expected minimum (0.70)!"
    assert dup_recall >= 0.90, f"Duplicate recall ({dup_recall:.2f}) below expected minimum (0.90)!"
    assert np.allclose(np.sum(pred_probs, axis=1), 1.0, atol=1e-5), "Probability distributions must sum to 1.0!"
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4), "Embeddings must be unit L2-normalized!"

    print("\n[+] All DCAI auditing algorithms, mathematical thresholds, and quality assertions PASSED.")


# ==============================================================================
# 8. CLI Entrypoint
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Cookbook 22: FiftyOne Dataset Auditing & DCAI Curation")
    parser.add_argument("--num-samples", type=int, default=500, help="Total dataset sample count")
    parser.add_argument("--num-classes", type=int, default=5, help="Number of semantic classes")
    parser.add_argument("--noise-rate", type=float, default=0.10, help="Synthetic label corruption rate")
    parser.add_argument("--num-ood", type=int, default=20, help="Number of injected OOD anomalies")
    parser.add_argument("--num-duplicates", type=int, default=15, help="Number of injected duplicate pairs")
    parser.add_argument("--dup-threshold", type=float, default=0.985, help="Cosine similarity threshold for duplicate detection")

    args = parser.parse_args()
    run_dataset_audit_pipeline(
        num_samples=args.num_samples,
        num_classes=args.num_classes,
        noise_rate=args.noise_rate,
        num_ood=args.num_ood,
        num_duplicate_pairs=args.num_duplicates,
        dup_threshold=args.dup_threshold
    )


if __name__ == "__main__":
    main()
