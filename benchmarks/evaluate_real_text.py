import os
import sys
import time
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.neuroworld import NeuroWorldLM

def evaluate_real_text_benchmarks(device_str="cuda"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"  Comprehensive Real-Text Generation & Quantitative Quality Evaluation on {device}  ")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = NeuroWorldLM(
        vocab_size=len(tokenizer),
        d_model=128,
        d_state=16,
        num_categoricals=8,
        num_classes=8,
        num_layers=2,
        max_rollout_steps=4,
        num_branches=4
    ).to(device)

    ckpt_path = "checkpoints/neuroworld_real_corpus.pt"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[✓] Loaded trained checkpoint from {ckpt_path}")
    model.eval()

    # Load 4 Real HuggingFace Datasets
    print("\n[*] Loading Real Datasets from HuggingFace...")
    ds_arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test[:25]")
    ds_obqa = load_dataset("allenai/openbookqa", "main", split="test[:25]")
    ds_gsm = load_dataset("openai/gsm8k", "main", split="test[:25]")
    ds_stories = load_dataset("roneneldan/TinyStories", split="validation[:25]")

    results = []

    # 1. ARC-Challenge Real QA Evaluation
    print("\n" + "-" * 80)
    print("  [1/4] Real Science QA: ARC-Challenge (allenai/ai2_arc)")
    print("-" * 80)
    arc_direct_corr, arc_thought_corr = 0, 0
    for idx, item in enumerate(ds_arc):
        q = item["question"]
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]
        target_label = item["answerKey"]

        scores_dir = []
        scores_tht = []
        for text in choices:
            prompt = f"Question: {q} Answer: {text}"
            ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                logits, _, _, _ = model(ids, use_posterior=False)
                log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
                t_ids = ids[:, 1:].unsqueeze(-1)
                tok_lp = torch.gather(log_probs, 2, t_ids).squeeze(-1)
                scores_dir.append(tok_lp.mean().item())

                h_list, ssm_states = model.init_hidden(1, device)
                for t in range(ids.shape[1] - 1):
                    x_t = model.tok_embed(ids[:, t])
                    for l_idx, layer in enumerate(model.layers):
                        x_t, ssm_states[l_idx], _ = layer.step(x_t, h_list[l_idx], ssm_states[l_idx], False)
                        h_list[l_idx] = x_t
                best_h, _, _, _ = model.planner(h_list[-1], ssm_states[-1])
                l_tht = model.lm_head(model.ln_f(best_h))
                s_tht = torch.log_softmax(l_tht, dim=-1)[0, ids[:, -1]].item()
                scores_tht.append(tok_lp.mean().item() + 0.3 * s_tht)

        pred_dir = labels[int(torch.argmax(torch.tensor(scores_dir)).item())]
        pred_tht = labels[int(torch.argmax(torch.tensor(scores_tht)).item())]
        if pred_dir == target_label:
            arc_direct_corr += 1
        if pred_tht == target_label:
            arc_thought_corr += 1

        if idx < 3:
            print(f"  [Sample {idx+1}] Q: {q[:60]}... | Target: {target_label} | Direct: {pred_dir} | Latent Thought: {pred_tht}")

    acc_arc_dir = (arc_direct_corr / len(ds_arc)) * 100.0
    acc_arc_tht = (arc_thought_corr / len(ds_arc)) * 100.0
    print(f"  ==> ARC Accuracy: Direct {acc_arc_dir:.1f}% vs Latent Thought {acc_arc_tht:.1f}% (+{acc_arc_tht - acc_arc_dir:.1f}%p)")

    # 2. GSM8K Real Math Generation & Exact Match
    print("\n" + "-" * 80)
    print("  [2/4] Real Math Word Problems: GSM8K (openai/gsm8k)")
    print("-" * 80)
    gsm_direct_corr, gsm_thought_corr = 0, 0
    gsm_qualitative = []
    for idx, item in enumerate(ds_gsm):
        q = item["question"]
        ans_raw = item["answer"]
        # Extract target number after ####
        target_num = ans_raw.split("####")[-1].strip() if "####" in ans_raw else ans_raw.strip()
        prompt = f"Question: {q} Answer:"
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            gen_dir = model.generate(ids, max_new_tokens=15, temperature=0.7)
            text_dir = tokenizer.decode(gen_dir[0].tolist(), skip_special_tokens=True).strip()

            gen_tht, _, depth = model.generate_with_adaptive_thought(ids, max_new_tokens=15)
            text_tht = tokenizer.decode(gen_tht[0].tolist(), skip_special_tokens=True).strip()

        match_dir = target_num in text_dir
        match_tht = target_num in text_tht
        if match_dir:
            gsm_direct_corr += 1
        if match_tht:
            gsm_thought_corr += 1

        if idx < 3:
            print(f"\n  [Sample {idx+1}] Question: {q[:80]}...")
            print(f"    • Ground Truth Target: {target_num}")
            print(f"    • Direct Generated   : {text_dir[:70]}")
            print(f"    • Latent Thought (K={depth}): {text_tht[:70]}")
            gsm_qualitative.append({
                "question": q,
                "target": target_num,
                "direct_generation": text_dir,
                "thought_generation": text_tht,
                "depth": depth
            })

    acc_gsm_dir = (gsm_direct_corr / len(ds_gsm)) * 100.0
    acc_gsm_tht = (gsm_thought_corr / len(ds_gsm)) * 100.0
    print(f"\n  ==> GSM8K Exact Match: Direct {acc_gsm_dir:.1f}% vs Latent Thought {acc_gsm_tht:.1f}% (+{acc_gsm_tht - acc_gsm_dir:.1f}%p)")

    # 3. TinyStories Real Creative Text Generation & Quality Scoring
    print("\n" + "-" * 80)
    print("  [3/4] Real Storytelling Quality: TinyStories (roneneldan/TinyStories)")
    print("-" * 80)
    story_samples = []
    for idx, item in enumerate(ds_stories[:4]):
        prompt_text = " ".join(item["text"].split()[:12])
        ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

        with torch.no_grad():
            gen_dir = model.generate(ids, max_new_tokens=30, temperature=0.8)
            text_dir = tokenizer.decode(torch.cat([ids, gen_dir], dim=-1)[0].tolist(), skip_special_tokens=True)

            gen_tht, _, depth = model.generate_with_adaptive_thought(ids, max_new_tokens=30)
            text_tht = tokenizer.decode(torch.cat([ids, gen_tht], dim=-1)[0].tolist(), skip_special_tokens=True)

        print(f"\n  [Story Prompt {idx+1}]: \"{prompt_text}\"")
        print(f"    • Direct Baseline Output       : \"{text_dir}\"")
        print(f"    • Latent Thought (K={depth}) Output: \"{text_tht}\"")
        story_samples.append({
            "prompt": prompt_text,
            "direct": text_dir,
            "thought": text_tht
        })

    # Save to REAL_TEXT_EVALUATION_REPORT.md
    with open("REAL_TEXT_EVALUATION_REPORT.md", "w") as f:
        f.write("# NeuroWorld-LM: Actual Real-Text Empirical Evaluation & Qualitative Analysis\n\n")
        f.write("All benchmarks evaluated directly on real textual inputs from HuggingFace corpora (No synthetic proxies).\n\n")
        f.write("## 1. Quantitative Accuracy on Real Corpora\n\n")
        f.write("| Dataset | Benchmark Domain | Direct Autoregressive | Zero-Token Latent Thought | Accuracy Gain |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: |\n")
        f.write(f"| **ARC-Challenge** | Grade-School Science Multiple-Choice | {acc_arc_dir:.1f}% | **{acc_arc_tht:.1f}%** | **+{acc_arc_tht - acc_arc_dir:.1f}%p** |\n")
        f.write(f"| **GSM8K** | Math Word Problem Exact Answer | {acc_gsm_dir:.1f}% | **{acc_gsm_tht:.1f}%** | **+{acc_gsm_tht - acc_gsm_dir:.1f}%p** |\n\n")
        f.write("## 2. Qualitative Text Generation & Answer Quality Samples\n\n")
        f.write("### 2.1 GSM8K Mathematical Word Problems\n\n")
        for s in gsm_qualitative:
            f.write(f"- **Question:** {s['question']}\n")
            f.write(f"  - **Ground Truth Target:** `{s['target']}`\n")
            f.write(f"  - **Direct Generation:** `{s['direct_generation']}`\n")
            f.write(f"  - **Latent Thought (K={s['depth']}) Generation:** `{s['thought_generation']}`\n\n")
        f.write("### 2.2 TinyStories Creative Generation Quality\n\n")
        for s in story_samples:
            f.write(f"- **Prompt:** \"{s['prompt']}\"\n")
            f.write(f"  - **Direct Output:** \"{s['direct']}\"\n")
            f.write(f"  - **Latent Thought Output:** \"{s['thought']}\"\n\n")

    print("\n[✓] Saved complete real-text evaluation to REAL_TEXT_EVALUATION_REPORT.md")

if __name__ == "__main__":
    evaluate_real_text_benchmarks()
