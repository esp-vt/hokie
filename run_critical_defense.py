import os
import torch
from models.neuroworld import NeuroWorldLM
from benchmarks.critical_defense_suite import CriticalDefenseBenchmark

def main():
    device = torch.device("cpu")
    print("=" * 70)
    print("  Executing Critical Defense Benchmarks (Refuting 3 Reviewer Criticisms)  ")
    print("=" * 70)

    vocab_size = 1024
    d_model = 128
    d_state = 16
    num_categoricals = 8
    num_classes = 8
    num_layers = 2

    model = NeuroWorldLM(
        vocab_size=vocab_size,
        d_model=d_model,
        d_state=d_state,
        num_categoricals=num_categoricals,
        num_classes=num_classes,
        num_layers=num_layers,
        max_rollout_steps=6,
        num_branches=4
    ).to(device)

    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "neuroworld_enhanced.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[✓] Loaded trained model from {ckpt_path}")
    else:
        print("[!] Warning: Checkpoint not found, using initialized model.")

    defense_suite = CriticalDefenseBenchmark(vocab_size=vocab_size, d_model=d_model)

    # 1. Deep Horizon Drift
    drift_res = defense_suite.evaluate_deep_horizon_drift(model, max_k_list=[1, 2, 4, 6, 8, 10], num_trials=32)

    # 2. Adversarial Noise Injection
    noise_res = defense_suite.evaluate_noise_robustness(model, noise_ratios=[0.0, 0.2, 0.4, 0.6, 0.8], num_trials=32)

    # 3. Strict FLOPs-to-Accuracy Pareto
    pareto_res = defense_suite.evaluate_flops_pareto(model, num_trials=32)

    print("\n" + "=" * 70)
    print("  All Critical Defense Benchmarks Finished! Writing Report...  ")
    print("=" * 70)

if __name__ == "__main__":
    main()
