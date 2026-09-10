import os
import time
import math
import torch
import torch.optim as optim
from data.hf_dataset_loader import RealCorpusDataLoader
from models.neuroworld import NeuroWorldLM

def evaluate_perplexity(model, loader, device, eval_iters=10):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = loader.get_batch("val")
            x, y = x.to(device), y.to(device)
            _, loss, metrics, _ = model(x, targets=y, use_posterior=False)
            total_loss += metrics["token_loss"]
    avg_loss = total_loss / eval_iters
    ppl = math.exp(min(10.0, avg_loss)) # Clamp for numerical display
    return avg_loss, ppl

def main():
    device = torch.device("cpu")
    print("=" * 75)
    print("  NeuroWorld-LM: Real HuggingFace Corpus Pre-Training & PPL Evaluation  ")
    print("=" * 75)

    seq_len = 64
    batch_size = 4
    total_steps = 60
    lr = 1e-3

    # Load Real HuggingFace WikiText Dataset with BPE Tokenizer
    loader = RealCorpusDataLoader(
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        tokenizer_name="gpt2",
        seq_len=seq_len,
        batch_size=batch_size
    )

    # Initialize NeuroWorld-LM with BPE Vocab
    d_model = 128
    d_state = 16
    num_categoricals = 8
    num_classes = 8
    num_layers = 2

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

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Initialized NeuroWorld-LM | Total Parameters: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    print(f"\n[*] Starting Pre-Training on Real WikiText-2 BPE Tokens ({total_steps} Steps)...")
    start_time = time.time()

    for step in range(1, total_steps + 1):
        model.train()
        optimizer.zero_grad()

        x, y = loader.get_batch("train")
        x, y = x.to(device), y.to(device)

        logits, loss, metrics, _ = model(x, targets=y, use_posterior=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 15 == 0 or step == 1:
            val_loss, val_ppl = evaluate_perplexity(model, loader, device, eval_iters=5)
            print(
                f"[Step {step:02d}/{total_steps}] "
                f"Train Loss: {metrics['token_loss']:.4f} | "
                f"KL: {metrics['kl_div']:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val PPL: {val_ppl:6.2f}"
            )

    elapsed = time.time() - start_time
    print("-" * 75)
    print(f"[✓] Real Corpus Training Finished in {elapsed:.2f}s")

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "neuroworld_real_corpus.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"[✓] Saved Real Corpus Model Checkpoint to: {ckpt_path}")

    # Text Generation Demonstration
    print("\n" + "=" * 75)
    print("  Real Text Autoregressive Generation Demonstration  ")
    print("=" * 75)
    prompt_text = "The discovery of the ancient world"
    prompt_ids = torch.tensor([loader.tokenizer.encode(prompt_text)], device=device)

    # 1. Direct Generation
    gen_ids_direct = model.generate(prompt_ids, max_new_tokens=15, temperature=0.8)
    full_text_direct = loader.tokenizer.decode(torch.cat([prompt_ids, gen_ids_direct], dim=-1)[0].tolist())
    print(f"[Prompt]: '{prompt_text}'")
    print(f"[Direct Generated Text]:\n  --> {full_text_direct}\n")

    # 2. Zero-Token Latent Rollout Generation
    gen_ids_thought, branch_scores, depth_k = model.generate_with_adaptive_thought(prompt_ids, max_new_tokens=15)
    full_text_thought = loader.tokenizer.decode(torch.cat([prompt_ids, gen_ids_thought], dim=-1)[0].tolist())
    print(f"[Zero-Token Latent Thought (K={depth_k}) Generated Text]:\n  --> {full_text_thought}\n")

if __name__ == "__main__":
    main()
