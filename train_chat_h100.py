#!/usr/bin/env python3
"""
NeuroWorld-LM: Conversational & Storytelling Pre-Training Pipeline on NVIDIA H100.
Trains NeuroWorldLM on natural language corpus (TinyStories / WikiText) with
Dual-Loop State Duality (SSM + RSSM) and Cognitive Active Forgetting (CAFE).
Saves checkpoint to checkpoints/neuroworld_chat.pt for interactive chat.
"""

import os
import sys
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.hf_dataset_loader import RealCorpusDataLoader
from models.neuroworld import NeuroWorldLM

def evaluate_perplexity(model, loader, device, eval_iters=10):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = loader.get_batch("val")
            x, y = x.to(device), y.to(device)
            _, _, metrics, _ = model(x, targets=y, use_posterior=False)
            total_loss += metrics["token_loss"]
    avg_loss = total_loss / eval_iters
    ppl = math.exp(min(10.0, avg_loss))
    return avg_loss, ppl

def parse_args():
    parser = argparse.ArgumentParser(description="NeuroWorld-LM Chat Pre-Training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--total_steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--output_ckpt", type=str, default="checkpoints/neuroworld_chat.pt")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 80)
    print("  NeuroWorld-LM: Conversational & Storytelling Pre-Training Engine  ")
    print(f"  Target Device: {device} | Total Steps: {args.total_steps} | Batch Size: {args.batch_size}")
    if torch.cuda.is_available() and "cuda" in args.device:
        print(f"  GPU Hardware: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    os.makedirs("checkpoints", exist_ok=True)

    # 1. Initialize Real Corpus DataLoader with GPT-2 Tokenizer
    print(f"\n[Step 1] Loading Corpus: {args.dataset}...")
    loader = RealCorpusDataLoader(
        dataset_name=args.dataset,
        dataset_config="",
        tokenizer_name="gpt2",
        seq_len=args.seq_len,
        batch_size=args.batch_size
    )

    # 2. Instantiate Enhanced NeuroWorldLM for Rich Conversational Generation
    print("\n[Step 2] Initializing NeuroWorldLM Architecture...")
    d_model = 256
    d_state = 32
    num_categoricals = 8
    num_classes = 8
    num_layers = 3

    model = NeuroWorldLM(
        vocab_size=loader.vocab_size,
        d_model=d_model,
        d_state=d_state,
        num_categoricals=num_categoricals,
        num_classes=num_classes,
        num_layers=num_layers,
        max_rollout_steps=4,
        num_branches=4
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  • Vocabulary Size         : {loader.vocab_size:,}")
    print(f"  • Hidden Dimension (d_model): {d_model}")
    print(f"  • SSM State Dimension (d_ssm): {d_state}")
    print(f"  • Categorical Latent Groups : {num_categoricals}x{num_classes}")
    print(f"  • Total Trainable Parameters: {total_params:,}")

    # 3. Optimizer & Learning Rate Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2, betas=(0.9, 0.95))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.total_steps, eta_min=args.lr * 0.1)

    # 4. Training Loop
    print(f"\n[Step 3] Launching Training on {device} ({args.total_steps} steps)...")
    start_time = time.time()
    best_val_loss = float("inf")

    for step in range(1, args.total_steps + 1):
        step_start = time.time()
        model.train()
        optimizer.zero_grad()

        x, y = loader.get_batch("train")
        x, y = x.to(device), y.to(device)

        logits, loss, metrics, _ = model(x, targets=y, use_posterior=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        step_elapsed = time.time() - step_start

        if step % 25 == 0 or step == 1:
            val_loss, val_ppl = evaluate_perplexity(model, loader, device, eval_iters=8)
            cur_lr = scheduler.get_last_lr()[0]
            print(
                f"[Step {step:03d}/{args.total_steps:03d}] "
                f"Train Loss: {metrics['token_loss']:.4f} | "
                f"KL Div: {metrics['kl_div']:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val PPL: {val_ppl:6.2f} | "
                f"Step: {step_elapsed*1000:.1f}ms | "
                f"LR: {cur_lr:.2e}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "step": step,
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "vocab_size": loader.vocab_size,
                    "d_model": d_model,
                    "d_state": d_state,
                    "num_layers": num_layers,
                    "num_categoricals": num_categoricals,
                    "num_classes": num_classes
                }, args.output_ckpt)

    total_elapsed = time.time() - start_time
    print("-" * 80)
    print(f"[✓] Training Completed in {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)!")
    print(f"[✓] Best Validation Loss: {best_val_loss:.4f} | Saved Checkpoint: {args.output_ckpt}")

    # 5. Quick Conversational Inference Verification
    print("\n" + "=" * 80)
    print("  Conversational Generation Sample Demonstration  ")
    print("=" * 80)
    test_prompts = [
        "Hello! How are you doing today?",
        "Once upon a time, a curious robot learned how to forget secrets"
    ]

    model.eval()
    for prompt in test_prompts:
        prompt_ids = torch.tensor([loader.tokenizer.encode(prompt)], device=device)
        with torch.no_grad():
            gen_ids = model.generate(prompt_ids, max_new_tokens=30, temperature=0.7)
            full_ids = torch.cat([prompt_ids, gen_ids], dim=-1)[0].tolist()
            decoded = loader.tokenizer.decode(full_ids)
        print(f"\n[Prompt]: {prompt}")
        print(f"[Response]:\n  --> {decoded}")

    print("\n[✓] All Setup & Checkpoint Verification Complete!")

if __name__ == "__main__":
    main()
