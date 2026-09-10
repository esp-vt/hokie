import os
import re
import time
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoTokenizer
from models.neuroworld import NeuroWorldLM

def extract_gsm8k_answer(text: str):
    """Extracts numeric answer following '####' or last integer in text."""
    match = re.search(r"####\s*(-?\d+)", text)
    if match:
        return int(match.group(1).replace(",", ""))
    digits = re.findall(r"-?\d+", text)
    return int(digits[-1]) if digits else None

class GSM8KDataLoader:
    def __init__(self, tokenizer_name="gpt2", max_samples=1000):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("[*] Loading GSM8K from HuggingFace (openai/gsm8k)...")
        raw_train = load_dataset("openai/gsm8k", "main", split=f"train[:{max_samples}]")
        raw_test = load_dataset("openai/gsm8k", "main", split="test[:100]")

        self.train_samples = self._process_dataset(raw_train)
        self.test_samples = self._process_dataset(raw_test)
        print(f"[✓] GSM8K Ready | Train: {len(self.train_samples)} | Test: {len(self.test_samples)}")

    def _process_dataset(self, dataset):
        samples = []
        for item in dataset:
            q = item["question"].strip()
            ans_num = extract_gsm8k_answer(item["answer"])
            if ans_num is not None:
                formatted = f"Question: {q} Answer: #### {ans_num}"
                token_ids = self.tokenizer.encode(formatted)
                q_ids = self.tokenizer.encode(f"Question: {q} Answer: ####")
                samples.append({
                    "token_ids": token_ids,
                    "q_ids": q_ids,
                    "target_num": ans_num,
                    "target_tok": self.tokenizer.encode(f" {ans_num}")[0]
                })
        return samples

    def get_batch(self, batch_size=8, seq_len=96):
        batch_items = [self.train_samples[i] for i in torch.randint(0, len(self.train_samples), (batch_size,))]
        inputs = []
        targets = []
        for item in batch_items:
            ids = item["token_ids"]
            if len(ids) > seq_len:
                ids = ids[:seq_len]
            else:
                ids = ids + [self.tokenizer.pad_token_id] * (seq_len - len(ids))
            
            in_t = torch.tensor(ids[:-1], dtype=torch.long)
            tar_t = torch.tensor(ids[1:], dtype=torch.long)
            inputs.append(in_t)
            targets.append(tar_t)
        return torch.stack(inputs), torch.stack(targets)

def main():
    device = torch.device("cpu")
    print("=" * 75)
    print("  Option C-2: GSM8K Natural Language Math Reasoning & Latent Rollout  ")
    print("=" * 75)

    loader = GSM8KDataLoader(max_samples=800)

    model = NeuroWorldLM(
        vocab_size=len(loader.tokenizer),
        d_model=128,
        d_state=16,
        num_categoricals=8,
        num_classes=8,
        num_layers=2,
        max_rollout_steps=4,
        num_branches=4
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-2)

    total_steps = 75
    print(f"\n[*] Fine-Tuning NeuroWorld-LM on GSM8K ({total_steps} steps)...")
    start_time = time.time()

    for step in range(1, total_steps + 1):
        model.train()
        optimizer.zero_grad()

        x, y = loader.get_batch(batch_size=8, seq_len=96)
        x, y = x.to(device), y.to(device)

        logits, loss, metrics, _ = model(x, targets=y, use_posterior=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 25 == 0 or step == 1:
            print(f"[GSM8K Step {step:02d}/{total_steps}] Train Loss: {metrics['token_loss']:.4f} | KL: {metrics['kl_div']:.4f}")

    elapsed = time.time() - start_time
    print(f"[✓] GSM8K Training Finished in {elapsed:.2f}s")
    
    ckpt_path = "checkpoints/neuroworld_gsm8k.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[✓] Saved GSM8K Model Checkpoint to: {ckpt_path}")

    # Evaluate on GSM8K Test Set: Direct vs Zero-Token Latent Rollout
    print("\n" + "=" * 75)
    print("  GSM8K Natural Language Mathematical Reasoning Benchmark  ")
    print("=" * 75)
    
    model.eval()
    num_eval_samples = 30
    eval_subset = loader.test_samples[:num_eval_samples]

    direct_correct = 0
    thought_correct = 0

    print(f"[*] Evaluating {num_eval_samples} Real English Math Word Problems...")
    for idx, sample in enumerate(eval_subset):
        q_tensor = torch.tensor([sample["q_ids"]], device=device)
        target_num = sample["target_num"]

        # 1. Direct Decoding
        with torch.no_grad():
            gen_direct = model.generate(q_tensor, max_new_tokens=5, temperature=0.7)
            decoded_direct = loader.tokenizer.decode(gen_direct[0].tolist())
            pred_num_direct = extract_gsm8k_answer("#### " + decoded_direct)

        if pred_num_direct == target_num:
            direct_correct += 1

        # 2. Zero-Token Latent Thought Rollout
        with torch.no_grad():
            gen_thought, _, _ = model.generate_with_adaptive_thought(q_tensor, max_new_tokens=5)
            decoded_thought = loader.tokenizer.decode(gen_thought[0].tolist())
            pred_num_thought = extract_gsm8k_answer("#### " + decoded_thought)

        if pred_num_thought == target_num:
            thought_correct += 1

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx+1}/{num_eval_samples} | Direct: {direct_correct}/{idx+1} | Latent Thought: {thought_correct}/{idx+1}")

    acc_direct = (direct_correct / num_eval_samples) * 100.0
    acc_thought = (thought_correct / num_eval_samples) * 100.0

    print("\n" + "-" * 75)
    print(f" [GSM8K Final Results]")
    print(f"  • Direct Autoregressive Accuracy : {acc_direct:5.1f}%")
    print(f"  • Zero-Token Latent Thought Acc  : {acc_thought:5.1f}%")
    print(f"  • Reasoning Gain (Accuracy Delta): {acc_thought - acc_direct:+5.1f}%")
    print("-" * 75)

if __name__ == "__main__":
    main()
