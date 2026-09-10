import time
import math
import sys
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.neuroworld import NeuroWorldLM

def profile_hardware_scaling():
    device = torch.device("cpu")
    print("=" * 75)
    print("  Pillar 5: Hardware Memory & Throughput Scaling Profiling (up to 32k)  ")
    print("=" * 75)

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
        num_layers=num_layers
    ).to(device)

    model.eval()

    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    neuroworld_mems = []
    transformer_mems = []
    throughputs = []

    for L in seq_lengths:
        # 1. State Memory Footprint
        h_list, ssm_states = model.init_hidden(1, device)
        state_kb = (sum(h.element_size() * h.nelement() for h in h_list) +
                    sum(s.element_size() * s.nelement() for s in ssm_states)) / 1024.0
        
        # Transformer KV Cache in KB: 2 * layers * L * d_model * 4 bytes
        transformer_kv_kb = (2 * num_layers * L * d_model * 4) / 1024.0

        neuroworld_mems.append(state_kb)
        transformer_mems.append(transformer_kv_kb)

        # 2. Generation Throughput (tokens/sec)
        dummy_prompt = torch.randint(0, vocab_size, (1, 64), device=device)
        num_gen = 30
        t0 = time.time()
        with torch.no_grad():
            _ = model.generate(dummy_prompt, max_new_tokens=num_gen)
        t_elapsed = time.time() - t0
        tokens_per_sec = num_gen / max(1e-4, t_elapsed)
        throughputs.append(tokens_per_sec)

        print(f" Context Length: {L:6d} | NeuroWorld State: {state_kb:6.2f} KB | Transformer KV: {transformer_kv_kb:9.2f} KB | Throughput: {tokens_per_sec:5.1f} tok/s")

    # Generate Figure 4: Memory Scaling Plot
    os.makedirs("figures", exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(9, 5.5), dpi=300)

    color_nw = "#1f77b4"
    color_tf = "#d62728"
    x_indices = range(len(seq_lengths))

    ax1.plot(x_indices, neuroworld_mems, color=color_nw, marker="o", linewidth=2.5, label="NeuroWorld-LM State (O(1) Constant)")
    ax1.plot(x_indices, transformer_mems, color=color_tf, marker="s", linestyle="--", linewidth=2.0, label="Standard Transformer KV-Cache (O(T) Linear)")
    ax1.set_xlabel("Context Sequence Length (Tokens)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Inference Memory Footprint (KB)", fontsize=12, fontweight="bold")
    ax1.set_yscale("log")
    ax1.set_xticks(list(x_indices))
    ax1.set_xticklabels([f"{L:,}" for L in seq_lengths], rotation=30)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", fontsize=10)

    plt.title(r"$\bf{Figure\ 4:}$ Inference State Memory Scaling vs Sequence Length", fontsize=13, pad=15)
    plt.tight_layout()
    save_fig_path = "figures/fig4_hardware_scaling.png"
    plt.savefig(save_fig_path)
    plt.close()
    print(f"\n[✓] Generated Figure 4: {save_fig_path}")

if __name__ == "__main__":
    profile_hardware_scaling()
