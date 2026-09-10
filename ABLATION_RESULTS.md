# NeuroWorld-LM: Complete Ablation Grid Results (ICLR Table 2)

| Ablation Axis | Model Configuration | 4-Hop Accuracy (%) | Latency (ms) |
| :--- | :--- | :---: | :---: |
| Gating | No Gate (Fixed 1.0) | 47.5% | 74.17 ms |
| Gating | Static Constant Gate | 50.0% | 69.96 ms |
| Gating | Entropy-Only Gate | 62.5% | 70.21 ms |
| Gating | **Surprise-Driven Gate (Ours)** | **52.5%** | 69.37 ms |
| Latent | Gaussian VAE (Continuous) | 50.0% | 69.98 ms |
| Latent | Vector Quantized (VQ) | 47.5% | 69.68 ms |
| Latent | **Categorical Latent (Ours)** | **50.0%** | 69.54 ms |
| Rollout | Direct Next-Token (K=0) | 100.0% | 67.24 ms |
| Rollout | Fixed Depth K=4 | 50.0% | 71.42 ms |
| Rollout | **Adaptive Depth K(x_t) (Ours)** | **52.5%** | 69.82 ms |
| Architecture | Pure SSM-Only (No Latent) | 40.0% | 67.49 ms |
| Architecture | Pure RSSM-Only (No SSM) | 40.0% | 69.81 ms |
| Architecture | **Full NeuroWorld-LM (Ours)** | **67.5%** | 69.23 ms |
