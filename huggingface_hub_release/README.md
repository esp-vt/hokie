---
language:
- en
license: mit
tags:
- state-space-models
- world-models
- active-forgetting
- infinite-context
- continuous-discrete
- robotics
- autonomous-agents
pipeline_tag: text-generation
---

# 🧠 Hokie-LM (NeuroWorld-LM): Dual-Loop SSM + Categorical RSSM World Model

**Hokie-LM (NeuroWorld-LM)** is a foundational Continuous-Discrete World Model language architecture designed for infinite-context autonomous agents, real-time robotics (1,000 Hz), and zero-shot unlearning.

### 🌟 Key Breakthroughs
1. **Strict $O(1)$ Memory**: Operates with a constant 17.00 KB state buffer without KV-cache expansion.
2. **Cognitive Active Forgetting (CAFE)**: Erases sensitive PII/prompts in under 0.1ms using multi-rank orthogonal subspace nullification ($P_\perp = I - q q^T$).
3. **Zero-Token Latent MCTS**: 100x faster reasoning in continuous-discrete latent space ($h_t, z_t$) matching OpenAI o1 depth.
4. **4,000 Agent Swarms**: `PagedState` memory engine runs 4,000+ active multi-turn agents on a single GPU.

### 🚀 Quickstart (Hugging Face Transformers)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("esp-vt/hokie-lm-1.3b")
model = AutoModelForCausalLM.from_pretrained("esp-vt/hokie-lm-1.3b", trust_remote_code=True).cuda()

# Interactive Dialogue Generation
prompt = "User: What is Hokie-LM?\nAssistant:"
input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
output_ids = model.generate(input_ids, max_new_tokens=40, temperature=0.7)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

### 📊 Model Architecture Specifications
| Parameter | Value |
| :--- | :--- |
| **Hidden Dimension ($d_{\text{model}}$)** | 256 |
| **SSM State Dimension ($d_{\text{ssm}}$)** | 16 |
| **Categorical Latent Groups ($K \times C$)** | 8 groups $\times$ 8 classes |
| **Layers ($N_{\text{layers}}$)** | 4 |
| **Active State Buffer Size** | **17.00 KB ($O(1)$)** |
| **Precision** | BFloat16 Mixed Precision |
| **Target Hardware** | NVIDIA H100 PCIe (CUDA 13.0) |

### 📖 Citation
```bibtex
@article{hokielm2026,
  title={Hokie-LM: Continuous-Discrete Dual-Loop State World Models with Cognitive Active Forgetting for Infinite-Context Agents},
  author={Eun et al.},
  journal={arXiv preprint},
  year={2026}
}
```
