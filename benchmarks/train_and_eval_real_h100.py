import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.neuroworld import NeuroWorldLM

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"  Real H100 GPU Training & Genuine Text Quality Evaluation on {device}  ")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Train on TinyStories on H100 GPU
    print("\n[Step 1/3] Training NeuroWorld-LM on Real TinyStories (Natural English Narrative)...")
    ds_stories = load_dataset("roneneldan/TinyStories", split="train[:5000]")
    
    all_story_ids = []
    for item in ds_stories:
        t = item["text"] if isinstance(item, dict) else str(item)
        if len(t.strip()) > 0:
            all_story_ids.extend(tokenizer.encode(t) + [tokenizer.eos_token_id])
    story_tensor = torch.tensor(all_story_ids, dtype=torch.long)
    print(f"[✓] Tokenized {len(story_tensor):,} real story tokens")

    model = NeuroWorldLM(
        vocab_size=len(tokenizer),
        d_model=256,
        d_state=16,
        num_categoricals=8,
        num_classes=8,
        num_layers=4,
        max_rollout_steps=4,
        num_branches=4
    ).to(device)

    # Use stable learning rate & weight decay for H100 training
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2, eps=1e-8)
    seq_len = 64
    batch_size = 16
    steps_story = 300

    start_t = time.time()
    for step in range(1, steps_story + 1):
        model.train()
        optimizer.zero_grad()
        starts = torch.randint(0, len(story_tensor) - seq_len - 1, (batch_size,))
        x = torch.stack([story_tensor[i : i + seq_len] for i in starts]).to(device)
        y = torch.stack([story_tensor[i + 1 : i + seq_len + 1] for i in starts]).to(device)

        logits, loss, metrics, _ = model(x, targets=y, use_posterior=True)
        if torch.isnan(loss):
            print(f"[!] Warning: NaN encountered at step {step}, skipping batch.")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 75 == 0 or step == 1:
            ppl = math.exp(min(10.0, metrics['token_loss']))
            print(f"  [TinyStories Step {step:03d}/{steps_story}] Loss: {metrics['token_loss']:.4f} | KL: {metrics['kl_div']:.4f} | PPL: {ppl:6.2f}")

    print(f"[✓] TinyStories Training finished in {time.time() - start_t:.2f}s")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/neuroworld_h100_stories.pt")

    # 2. Fine-tune on GSM8K Math Problems on H100
    print("\n[Step 2/3] Fine-Tuning NeuroWorld-LM on Real GSM8K Math Word Problems...")
    ds_gsm_train = load_dataset("openai/gsm8k", "main", split="train[:1500]")
    gsm_samples = []
    for item in ds_gsm_train:
        q = item["question"].strip()
        a = item["answer"].strip()
        ans_target = a.split("####")[-1].strip() if "####" in a else a
        formatted = f"Question: {q}\nAnswer: #### {ans_target}"
        ids = tokenizer.encode(formatted)
        if len(ids) <= 128:
            gsm_samples.append(ids)

    print(f"[✓] Prepared {len(gsm_samples)} real GSM8K math samples")
    steps_gsm = 250
    start_t = time.time()
    optimizer_gsm = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    for step in range(1, steps_gsm + 1):
        model.train()
        optimizer_gsm.zero_grad()
        batch_items = [gsm_samples[i] for i in torch.randint(0, len(gsm_samples), (batch_size,))]
        max_l = max(len(s) for s in batch_items)
        padded = [s + [tokenizer.pad_token_id] * (max_l - len(s)) for s in batch_items]
        p_tensor = torch.tensor(padded, dtype=torch.long, device=device)
        x = p_tensor[:, :-1]
        y = p_tensor[:, 1:]

        logits, loss, metrics, _ = model(x, targets=y, use_posterior=True)
        if torch.isnan(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_gsm.step()

        if step % 50 == 0 or step == 1:
            print(f"  [GSM8K Step {step:03d}/{steps_gsm}] Loss: {metrics['token_loss']:.4f} | KL: {metrics['kl_div']:.4f}")

    print(f"[✓] GSM8K Training finished in {time.time() - start_t:.2f}s")
    torch.save(model.state_dict(), "checkpoints/neuroworld_h100_gsm.pt")

    # 3. Genuine Real-Text Output Generation & Comparison
    print("\n" + "=" * 80)
    print("  [Step 3/3] Genuine Real Text Generation & Quality Inspection  ")
    print("=" * 80)
    model.eval()

    # Test 1: Real Story Prompts
    story_prompts = [
        "Once upon a time, a little girl named Lily found a magic key in the garden.",
        "One day, a friendly dog saw a lost kitten and decided to help.",
        "Tom wanted to build a big toy castle with his blocks."
    ]
    print("\n--- [A] Genuine TinyStories Story Continuations ---")
    for p in story_prompts:
        p_ids = tokenizer.encode(p, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_dir = model.generate(p_ids, max_new_tokens=25, temperature=0.7)
            out_dir = tokenizer.decode(torch.cat([p_ids, gen_dir], dim=-1)[0].tolist(), skip_special_tokens=True)

            gen_tht, _, depth = model.generate_with_adaptive_thought(p_ids, max_new_tokens=25)
            out_tht = tokenizer.decode(torch.cat([p_ids, gen_tht], dim=-1)[0].tolist(), skip_special_tokens=True)

        print(f"\n[Prompt]: \"{p}\"")
        print(f"  --> Direct Autoregressive : \"{out_dir}\"")
        print(f"  --> Latent Thought (K={depth}): \"{out_tht}\"")

    # Test 2: Real GSM8K Math Problems
    ds_gsm_test = load_dataset("openai/gsm8k", "main", split="test[:10]")
    print("\n--- [B] Genuine GSM8K Math Problem Generations ---")
    gsm_correct_dir, gsm_correct_tht = 0, 0
    for idx, item in enumerate(ds_gsm_test):
        q = item["question"]
        target = item["answer"].split("####")[-1].strip() if "####" in item["answer"] else item["answer"].strip()
        p_ids = tokenizer.encode(f"Question: {q}\nAnswer: ####", return_tensors="pt").to(device)

        with torch.no_grad():
            gen_dir = model.generate(p_ids, max_new_tokens=8, temperature=0.5)
            out_dir = tokenizer.decode(gen_dir[0].tolist(), skip_special_tokens=True).strip()

            gen_tht, _, depth = model.generate_with_adaptive_thought(p_ids, max_new_tokens=8)
            out_tht = tokenizer.decode(gen_tht[0].tolist(), skip_special_tokens=True).strip()

        match_dir = target in out_dir
        match_tht = target in out_tht
        if match_dir:
            gsm_correct_dir += 1
        if match_tht:
            gsm_correct_tht += 1

        if idx < 3:
            print(f"\n[Problem {idx+1}]: {q[:90]}...")
            print(f"  • Ground Truth Answer: {target}")
            print(f"  • Direct Generated   : {out_dir}")
            print(f"  • Latent Thought (K={depth}): {out_tht}")

    print("\n" + "=" * 80)
    print(f"  [Real H100 Evaluation Summary]")
    print(f"  • GSM8K Direct Accuracy : {gsm_correct_dir}/{len(ds_gsm_test)} ({gsm_correct_dir*10:.1f}%)")
    print(f"  • GSM8K Thought Accuracy: {gsm_correct_tht}/{len(ds_gsm_test)} ({gsm_correct_tht*10:.1f}%)")
    print("=" * 80)

if __name__ == "__main__":
    main()
