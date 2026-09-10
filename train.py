import os
import time
import torch
import torch.optim as optim
from models.neuroworld import NeuroWorldLM
from benchmarks.synthetic_tasks import SyntheticTaskGenerator

def main():
    print("=" * 60)
    print("NeuroWorld-LM Training & State Verification Engine (CPU)")
    print("=" * 60)

    device = torch.device("cpu")
    vocab_size = 512
    d_model = 128
    d_latent = 32
    d_state = 16
    num_layers = 2
    batch_size = 16
    total_steps = 150
    lr = 2e-3

    # Initialize model
    model = NeuroWorldLM(
        vocab_size=vocab_size,
        d_model=d_model,
        d_latent=d_latent,
        d_state=d_state,
        num_layers=num_layers,
        rollout_steps=3,
        num_branches=3
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Model initialized | Total Parameters: {total_params:,}")
    print(f"[*] Hardware: {device} | Fast CPU Verification Mode")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    task_gen = SyntheticTaskGenerator(vocab_size=vocab_size)

    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "neuroworld_best.pt")
    start_time = time.time()

    for step in range(1, total_steps + 1):
        model.train()
        optimizer.zero_grad()

        # Alternate between Associative Recall and World State Tracking
        if step % 2 == 0:
            inputs, targets = task_gen.generate_associative_recall_batch(batch_size=batch_size, num_pairs=4, noise_len=24)
            task_name = "Associative Recall"
        else:
            inputs, targets = task_gen.generate_state_tracking_batch(batch_size=batch_size, num_steps=4)
            task_name = "State Tracking"

        inputs, targets = inputs.to(device), targets.to(device)
        logits, loss, metrics, _ = model(inputs, targets=targets, use_posterior=True)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 25 == 0 or step == 1:
            # Measure target token accuracy (last position)
            with torch.no_grad():
                preds = torch.argmax(logits[:, -1, :], dim=-1)
                correct = (preds == targets[:, -1]).float().mean().item() * 100.0

            print(
                f"[Step {step:03d}/{total_steps}] Task: {task_name:<18} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"KL Div: {metrics['kl_div']:.4f} | "
                f"Surprise: {metrics['mean_surprise']:.4f} | "
                f"Query Acc: {correct:5.1f}%"
            )

    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"[✓] Training completed in {elapsed:.2f}s | Saving checkpoint...")
    torch.save(model.state_dict(), ckpt_path)
    print(f"[✓] Saved model checkpoint to {ckpt_path}")

if __name__ == "__main__":
    main()
