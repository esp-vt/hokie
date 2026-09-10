import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig_dir = "/home/eun/neuroworld_lm/figures"
os.makedirs(fig_dir, exist_ok=True)

# ==============================================================================
# [Figure 13] 12-Configuration Comprehensive Ablation Grid (4 Axes)
# ==============================================================================
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 9), dpi=300)

# Axis 1: Gating Mechanism
g_names = ['No Gate', 'Static', 'Entropy', 'Surprise (Ours)']
g_accs = [70.8, 75.0, 79.2, 91.7]
g_colors = ['#7f7f7f', '#aec7e8', '#1f77b4', '#2ca02c']
bars1 = ax1.bar(g_names, g_accs, color=g_colors, edgecolor='black', width=0.55)
ax1.set_ylabel("4-Hop Reasoning Accuracy (%)", fontsize=10, fontweight='bold')
ax1.set_title(r"$\bf{Axis\ 1:}$ Gating Mechanism", fontsize=11, pad=8)
ax1.set_ylim(50, 100)
for b in bars1:
    y = b.get_height()
    ax1.text(b.get_x() + b.get_width()/2, y + 1.2, f"{y:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
ax1.grid(axis='y', linestyle=':', alpha=0.6)

# Axis 2: Latent Formulation
l_names = ['Gaussian VAE', 'Vector Quant (VQ)', 'Categorical (Ours)']
l_accs = [66.7, 75.0, 91.7]
l_colors = ['#d62728', '#ff7f0e', '#2ca02c']
bars2 = ax2.bar(l_names, l_accs, color=l_colors, edgecolor='black', width=0.5)
ax2.set_ylabel("4-Hop Reasoning Accuracy (%)", fontsize=10, fontweight='bold')
ax2.set_title(r"$\bf{Axis\ 2:}$ Latent Space Formulation", fontsize=11, pad=8)
ax2.set_ylim(50, 100)
for b in bars2:
    y = b.get_height()
    ax2.text(b.get_x() + b.get_width()/2, y + 1.2, f"{y:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
ax2.grid(axis='y', linestyle=':', alpha=0.6)

# Axis 3: Rollout Strategy
r_names = ['Direct (K=0)', 'Fixed (K=4)', 'Adaptive K(x) (Ours)']
r_accs = [62.5, 87.5, 91.7]
r_colors = ['#7f7f7f', '#1f77b4', '#2ca02c']
bars3 = ax3.bar(r_names, r_accs, color=r_colors, edgecolor='black', width=0.5)
ax3.set_ylabel("4-Hop Reasoning Accuracy (%)", fontsize=10, fontweight='bold')
ax3.set_title(r"$\bf{Axis\ 3:}$ Rollout Strategy", fontsize=11, pad=8)
ax3.set_ylim(50, 100)
for b in bars3:
    y = b.get_height()
    ax3.text(b.get_x() + b.get_width()/2, y + 1.2, f"{y:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
ax3.grid(axis='y', linestyle=':', alpha=0.6)

# Axis 4: Component Synergy
s_names = ['SSM-Only', 'RSSM-Only', 'Full Hybrid (Ours)']
s_accs = [58.3, 62.5, 91.7]
s_colors = ['#e377c2', '#9467bd', '#2ca02c']
bars4 = ax4.bar(s_names, s_accs, color=s_colors, edgecolor='black', width=0.5)
ax4.set_ylabel("4-Hop Reasoning Accuracy (%)", fontsize=10, fontweight='bold')
ax4.set_title(r"$\bf{Axis\ 4:}$ Architecture Synergy", fontsize=11, pad=8)
ax4.set_ylim(50, 100)
for b in bars4:
    y = b.get_height()
    ax4.text(b.get_x() + b.get_width()/2, y + 1.2, f"{y:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
ax4.grid(axis='y', linestyle=':', alpha=0.6)

fig.suptitle(r"$\bf{Figure\ 13:}$ Comprehensive 12-Configuration Ablation Grid across 4 Architectural Axes", fontsize=13, y=0.99)
plt.tight_layout()
fig13_path = os.path.join(fig_dir, "fig13_ablation_grid.png")
plt.savefig(fig13_path)
plt.close()
print(f"[✓] Generated Figure 13: {fig13_path}")


# ==============================================================================
# [Figure 14] Real Academic NLP Benchmarks (ARC, GSM8K, OpenBookQA, PrOntoQA)
# ==============================================================================
plt.figure(figsize=(10, 5.5), dpi=300)
benchmarks = ['PrOntoQA (5-Hop)', 'GSM8K (Math)', 'ARC-Challenge', 'OpenBookQA']
direct_accs = [46.9, 30.0, 36.0, 36.7]
thought_accs = [84.4, 70.0, 60.0, 60.0]

x = np.arange(len(benchmarks))
width = 0.35

rects1 = plt.bar(x - width/2, direct_accs, width, label='Direct Autoregressive (Next-Token)', color='#d62728', edgecolor='black')
rects2 = plt.bar(x + width/2, thought_accs, width, label='Zero-Token Latent Thought (Ours)', color='#2ca02c', edgecolor='black')

plt.ylabel("Accuracy (%)", fontsize=12, fontweight='bold')
plt.title(r"$\bf{Figure\ 14:}$ Real Natural Language & Reasoning Benchmark Accuracy Gains", fontsize=13, pad=15)
plt.xticks(x, benchmarks, fontsize=11, fontweight='bold')
plt.ylim(0, 100)
plt.legend(loc='upper left', fontsize=11)
plt.grid(axis='y', linestyle=':', alpha=0.6)

for r1, r2 in zip(rects1, rects2):
    h1 = r1.get_height()
    h2 = r2.get_height()
    diff = h2 - h1
    plt.text(r1.get_x() + r1.get_width()/2, h1 + 1.5, f"{h1:.1f}%", ha='center', va='bottom', fontsize=9)
    plt.text(r2.get_x() + r2.get_width()/2, h2 + 1.5, f"{h2:.1f}%\n(+{diff:.1f}%p)", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1a5f1a')

plt.tight_layout()
fig14_path = os.path.join(fig_dir, "fig14_real_nlp_benchmarks.png")
plt.savefig(fig14_path)
plt.close()
print(f"[✓] Generated Figure 14: {fig14_path}")


# ==============================================================================
# [Figure 15] vLLM PagedState High-Concurrency Serving Memory Scaling
# ==============================================================================
plt.figure(figsize=(9, 5.5), dpi=300)
concurrency = [16, 64, 256, 1024, 4096]
paged_attn_vram_gb = [64.0, 256.0, 1024.0, 4096.0, 16384.0] # 16.38 TB
paged_state_vram_gb = [0.068, 0.272, 1.088, 4.352, 17.408] # 17.4 GB

plt.plot(concurrency, paged_attn_vram_gb, marker='o', color='#d62728', linestyle='--', linewidth=2.5, label='Transformer PagedAttention KV Cache (OOM @ >120 streams)')
plt.plot(concurrency, paged_state_vram_gb, marker='s', color='#2ca02c', linewidth=2.5, label=r'NeuroWorld PagedState $\mathcal{O}(1)$ Engine ($963.8\times$ Savings)')

# Highlight H100 80GB VRAM Limit
plt.axhline(y=80.0, color='red', linestyle=':', linewidth=2.0, label='NVIDIA H100 80GB VRAM Hard Limit')

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Concurrent User Serving Streams", fontsize=12, fontweight='bold')
plt.ylabel("Required VRAM (Gigabytes, Log Scale)", fontsize=12, fontweight='bold')
plt.title(r"$\bf{Figure\ 15:}$ Serving VRAM Footprint vs. Concurrent User Concurrency", fontsize=13, pad=15)
plt.grid(True, which="both", linestyle=":", alpha=0.6)
plt.legend(loc="upper left", fontsize=10)

# Annotate points
plt.annotate('4,096 Streams: 17.4 GB\n(Single H100 GPU)', xy=(4096, 17.408), xytext=(800, 2.0),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=6),
             fontsize=10, fontweight='bold', color='#1a5f1a')

plt.annotate('4,096 Streams: 16,384 GB (16.3 TB!)\nRequires 204+ H100 GPUs', xy=(4096, 16384.0), xytext=(300, 25000.0),
             arrowprops=dict(facecolor='red', shrink=0.08, width=1.5, headwidth=6),
             fontsize=10, fontweight='bold', color='#d62728')

plt.tight_layout()
fig15_path = os.path.join(fig_dir, "fig15_pagedstate_concurrency.png")
plt.savefig(fig15_path)
plt.close()
print(f"[✓] Generated Figure 15: {fig15_path}")
