import os
import time
import torch
import torch.optim as optim
from models.neuroworld import NeuroWorldLM
from benchmarks.synthetic_tasks import SyntheticTaskGenerator
from benchmarks.mqar_eval import MQARBenchmark
from benchmarks.prontoqa_gsm_eval import ReasoningProtocolBenchmark

def train_enhanced_neuroworld(device, steps=200):
    print("=" * 70)
    print("  Phase 1 & 2: Training NeuroWorld-LM with Categorical RSSM & SSD Scan  ")
    print("=" * 70)

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
        num_layers=num_layers,
        max_rollout_steps=4,
        num_branches=4
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    task_gen = SyntheticTaskGenerator(vocab_size=vocab_size)

    start_time = time.time()
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad()

        # Multi-task training mixture: MQAR, State Tracking, Logic
        if step % 2 == 0:
            inputs, targets = task_gen.generate_associative_recall_batch(batch_size=16, num_pairs=6, noise_len=32)
        else:
            inputs, targets = task_gen.generate_state_tracking_batch(batch_size=16, num_steps=5)

        inputs, targets = inputs.to(device), targets.to(device)
        logits, loss, metrics, _ = model(inputs, targets=targets, use_posterior=True)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == 1:
            preds = torch.argmax(logits[:, -1, :], dim=-1)
            acc = (preds == targets[:, -1]).float().mean().item() * 100.0
            print(f"[Train Step {step:03d}/{steps}] Loss: {metrics['loss']:.4f} | KL: {metrics['kl_div']:.4f} | Surprise: {metrics['mean_surprise']:.4f} | Acc: {acc:5.1f}%")

    elapsed = time.time() - start_time
    print(f"[✓] Enhanced Model Training finished in {elapsed:.2f}s")
    
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "neuroworld_enhanced.pt"))
    return model

def main():
    device = torch.device("cpu")
    print(f"[*] Running Academic Benchmark Suite on: {device}")

    # 1. Train Model with Categorical RSSM & SSD Scan
    model = train_enhanced_neuroworld(device, steps=200)

    # 2. Benchmark MQAR at Scale (up to 8k & 16k)
    mqar = MQARBenchmark(vocab_size=1024)
    mqar_results = mqar.evaluate_model_on_lengths(
        model,
        lengths=[512, 1024, 2048, 4096, 8192, 16384],
        num_trials=16
    )

    # 3. Benchmark Multi-hop Reasoning (PrOntoQA & Symbolic GSM)
    reasoning_bench = ReasoningProtocolBenchmark(vocab_size=1024)
    reasoning_bench.evaluate_reasoning_suite(model, num_samples=32)

    print("\n" + "=" * 70)
    print("  Academic Paper Benchmark Evaluation Successfully Completed!  ")
    print("=" * 70)

if __name__ == "__main__":
    main()
