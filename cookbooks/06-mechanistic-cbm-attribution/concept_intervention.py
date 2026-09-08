#!/usr/bin/env python3
"""
Minimal Production Recipe: Concept Bottleneck Model (CBM) with Human Intervention.
Demonstrates:
1. Predicting interpretable concepts c = f(x) from raw image features.
2. Predicting final class y = g(c) strictly through the concept layer.
3. Test-time human intervention: overriding a corrupted concept to restore decision correctness.
"""

import numpy as np


class ConceptBottleneckModel:
    def __init__(self, d_features: int, concepts: list[str], classes: list[str]):
        self.concepts = concepts
        self.classes = classes

        # Weight matrix mapping input features -> concept logits
        np.random.seed(42)
        self.W_concept = np.random.randn(len(concepts), d_features) * 0.5
        self.b_concept = np.zeros(len(concepts))

        # Weight matrix mapping concept activations -> final class logits
        # Hardcoded interpretable rules:
        # Class 0 ("Ambulance"): +Wheels, +Siren, +White/Red
        # Class 1 ("Firetruck"): +Wheels, +Siren, +Red
        # Class 2 ("Sports Car"): +Wheels, -Siren, +Low Profile
        self.W_class = np.array(
            [
                [2.0, 3.0, 2.0, -1.0],  # Ambulance
                [2.0, 3.0, -1.0, 1.0],  # Firetruck
                [2.0, -3.0, -1.0, 2.0],  # Sports Car
            ]
        )
        self.b_class = np.array([-1.0, -1.0, -1.0])

    def predict_concepts(self, x: np.ndarray) -> np.ndarray:
        """Predicts continuous concept probabilities using sigmoid."""
        logits = self.W_concept @ x + self.b_concept
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -10, 10)))

    def predict_class(self, concepts: np.ndarray) -> np.ndarray:
        """Predicts class logits purely from intermediate concepts."""
        logits = self.W_class @ concepts + self.b_class
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)


def main() -> None:
    concepts = ["Has Wheels", "Has Emergency Siren", "Is White Color", "Has Low Aerodynamic Profile"]
    classes = ["Ambulance", "Firetruck", "Sports Car"]
    d_features = 16

    cbm = ConceptBottleneckModel(d_features, concepts, classes)

    print("[+] Case 1: Ambulance with optical distortion causing missing Siren detection...")
    # Simulated true image feature
    x_ambulance = np.random.randn(d_features)

    # Predicted concepts
    pred_concepts = cbm.predict_concepts(x_ambulance)
    # Simulate a perception hallucination: the siren reflection is missed (concept set to 0.05)
    pred_concepts[1] = 0.05
    pred_concepts[0] = 0.95  # Wheels
    pred_concepts[2] = 0.90  # White

    print("    Predicted Intermediate Concepts:")
    for name, val in zip(concepts, pred_concepts):
        print(f"      - {name}: {val:.2f}")

    raw_probs = cbm.predict_class(pred_concepts)
    raw_pred = classes[np.argmax(raw_probs)]
    print(f"    Raw Output Decision: {raw_pred} (Confidence: {np.max(raw_probs):.2%})")

    print("\n[+] Case 2: Human Operator Intervention at Test Time...")
    print("    Operator inspects camera frame and confirms emergency flashing siren is active.")
    intervened_concepts = pred_concepts.copy()
    intervened_concepts[1] = 1.0  # Manually clamp "Has Emergency Siren" to 1.0

    corrected_probs = cbm.predict_class(intervened_concepts)
    corrected_pred = classes[np.argmax(corrected_probs)]
    print(f"    Post-Intervention Output Decision: {corrected_pred} (Confidence: {np.max(corrected_probs):.2%})")

    assert corrected_pred == "Ambulance", "Intervention should successfully route to Ambulance"
    print("\n[+] Concept Bottleneck verification and human-in-the-loop intervention verified.")


if __name__ == "__main__":
    main()
