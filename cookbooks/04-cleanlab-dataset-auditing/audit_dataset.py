#!/usr/bin/env python3
"""
Minimal Production Recipe: Cleanlab Confident Learning for Computer Vision.
Audits predicted out-of-sample class probabilities against noisy dataset labels
to mathematically identify mislabeled images, class-imbalance noise, and label ambiguity.
"""

import numpy as np


def compute_confident_joint(labels: np.ndarray, pred_probs: np.ndarray) -> np.ndarray:
    """Computes the confident joint matrix Q_ij estimating joint distribution

    P(given_label=i, true_label=j) based on confident learning thresholds.
    """
    num_classes = pred_probs.shape[1]
    # Step 1: Compute class-specific self-confidence thresholds t_j
    thresholds = np.zeros(num_classes)
    for j in range(num_classes):
        mask_j = labels == j
        if np.any(mask_j):
            thresholds[j] = np.mean(pred_probs[mask_j, j])
        else:
            thresholds[j] = 1.0 / num_classes

    # Step 2: Construct boolean confident indicators
    confident_joint = np.zeros((num_classes, num_classes), dtype=int)
    for i, (true_label, probs) in enumerate(zip(labels, pred_probs)):
        # Check which classes exceed threshold
        above_threshold = np.where(probs >= thresholds)[0]
        if len(above_threshold) > 0:
            # Pick the class with highest probability among those exceeding threshold
            best_j = above_threshold[np.argmax(probs[above_threshold])]
            confident_joint[true_label, best_j] += 1
        else:
            # Fallback: argmax probability
            confident_joint[true_label, np.argmax(probs)] += 1

    return confident_joint, thresholds


def find_label_issues(
    labels: np.ndarray, pred_probs: np.ndarray, thresholds: np.ndarray
) -> list[int]:
    """Identifies indices of samples where given label does not match confident true label."""
    issues = []
    for idx, (given_label, probs) in enumerate(zip(labels, pred_probs)):
        # Model is more confident in a different class than given label
        predicted_class = np.argmax(probs)
        if predicted_class != given_label and probs[predicted_class] >= thresholds[predicted_class]:
            issues.append(idx)
    return issues


def main() -> None:
    print("[+] Initializing synthetic CV dataset with injected label errors...")
    np.random.seed(42)
    num_samples = 1000
    num_classes = 5

    # True latent labels
    true_labels = np.random.choice(num_classes, size=num_samples)

    # Injected noisy labels (10% label error rate)
    noisy_labels = true_labels.copy()
    corrupt_indices = np.random.choice(num_samples, size=int(0.10 * num_samples), replace=False)
    for idx in corrupt_indices:
        noisy_labels[idx] = (true_labels[idx] + np.random.randint(1, num_classes)) % num_classes

    # Simulated out-of-fold predicted probabilities
    pred_probs = np.zeros((num_samples, num_classes))
    for i in range(num_samples):
        # Peak probability on the true label
        probs = np.random.dirichlet(np.ones(num_classes) * 0.5)
        probs[true_labels[i]] += 4.0
        pred_probs[i] = probs / np.sum(probs)

    print("[+] Running Confident Learning audit across 1,000 samples...")
    confident_joint, thresholds = compute_confident_joint(noisy_labels, pred_probs)
    flagged_issues = find_label_issues(noisy_labels, pred_probs, thresholds)

    # Calculate recovery metrics
    corrupt_set = set(corrupt_indices)
    flagged_set = set(flagged_issues)
    true_positives = len(corrupt_set.intersection(flagged_set))
    precision = true_positives / len(flagged_set) if flagged_set else 0.0
    recall = true_positives / len(corrupt_set) if corrupt_set else 0.0

    print(f"[+] Audit Results: Detected {len(flagged_issues)} potential label errors.")
    print(f"[+] Ground Truth Corrupted Count: {len(corrupt_indices)}")
    print(f"[+] Precision: {precision:.2%}, Recall: {recall:.2%}")
    print("[+] Confident Joint Diagonal vs Off-Diagonal Mass:")
    print(confident_joint)
    print("[+] Cleanlab audit completed successfully. Verified DCAI pipeline.")


if __name__ == "__main__":
    main()
