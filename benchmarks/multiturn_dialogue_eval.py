import time
import random
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.neuroworld import NeuroWorldLM

class MultiTurnDialogueBenchmark:
    """
    Simulates and trains for long-horizon multi-turn conversations (20 ~ 100 turns).
    """
    def __init__(self, vocab_size: int = 1024):
        self.vocab_size = vocab_size
        self.bos = 1
        self.query_tok = 3
        self.assign_tok = 4
        self.fact_key_offset = 15
        self.fact_val_offset = 210
        self.filler_offset = 500

    def generate_dialogue_batch(self, batch_size: int = 16, num_turns: int = 50):
        inputs = []
        targets = []

        for _ in range(batch_size):
            secret_key = random.randint(self.fact_key_offset, self.fact_key_offset + 30)
            secret_val = random.randint(self.fact_val_offset, self.fact_val_offset + 50)
            
            seq = [self.bos, secret_key, self.assign_tok, secret_val]

            for _ in range(2, num_turns):
                filler_len = random.randint(2, 4)
                filler_tokens = [random.randint(self.filler_offset, self.vocab_size - 1) for _ in range(filler_len)]
                seq.extend(filler_tokens)

            seq.extend([self.query_tok, secret_key])
            
            input_t = torch.tensor(seq, dtype=torch.long)
            target_t = torch.tensor(seq[1:] + [secret_val], dtype=torch.long)
            inputs.append(input_t)
            targets.append(target_t)

        max_len = max(x.size(0) for x in inputs)
        padded_inputs = torch.full((batch_size, max_len), 0, dtype=torch.long)
        padded_targets = torch.full((batch_size, max_len), 0, dtype=torch.long)
        for i in range(batch_size):
            padded_inputs[i, :inputs[i].size(0)] = inputs[i]
            padded_targets[i, :targets[i].size(0)] = targets[i]

        return padded_inputs, padded_targets

    def evaluate(self, model: nn.Module, num_turns: int = 100, num_trials: int = 10):
        model.eval()
        device = next(model.parameters()).device
        correct = 0

        print(f"\n[Multi-Turn Evaluation] Testing {num_trials} Conversations at {num_turns}-Turn Horizon")
        print("-" * 70)

        for trial_idx in range(1, num_trials + 1):
            secret_key = random.randint(self.fact_key_offset, self.fact_key_offset + 30)
            secret_val = random.randint(self.fact_val_offset, self.fact_val_offset + 50)
            
            seq = [self.bos, secret_key, self.assign_tok, secret_val]
            for _ in range(2, num_turns):
                filler_len = random.randint(2, 4)
                filler_tokens = [random.randint(self.filler_offset, self.vocab_size - 1) for _ in range(filler_len)]
                seq.extend(filler_tokens)

            seq.extend([self.query_tok, secret_key])
            input_tensor = torch.tensor([seq], dtype=torch.long, device=device)

            with torch.no_grad():
                logits, _, _, _ = model(input_tensor, use_posterior=False)
                predicted_val = torch.argmax(logits[:, -1, :], dim=-1).item()

            is_match = (predicted_val == secret_val)
            if is_match:
                correct += 1
            print(f" Trial {trial_idx:02d} | Expected: {secret_val}, Got: {predicted_val} | Match: {is_match}")

        acc = (correct / num_trials) * 100.0
        print(f"\n[Multi-Turn {num_turns}-Turn Retention Accuracy]: {correct}/{num_trials} ({acc:.1f}%)")
        return acc

def main():
    device = torch.device("cpu")
    vocab_size = 1024
    model = NeuroWorldLM(
        vocab_size=vocab_size,
        d_model=128,
        d_state=16,
        num_categoricals=8,
        num_classes=8,
        num_layers=2
    ).to(device)

    bench = MultiTurnDialogueBenchmark(vocab_size=vocab_size)
    optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)

    print("=" * 70)
    print("  Training & Evaluating NeuroWorld-LM on Long Multi-Turn Dialogues  ")
    print("=" * 70)

    # Multi-turn curriculum training
    for step in range(1, 101):
        model.train()
        optimizer.zero_grad()
        turns = random.randint(20, 60)
        inputs, targets = bench.generate_dialogue_batch(batch_size=16, num_turns=turns)
        inputs, targets = inputs.to(device), targets.to(device)
        logits, loss, metrics, _ = model(inputs, targets=targets, use_posterior=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 25 == 0:
            print(f"[Train Step {step:03d}/100] Loss: {metrics['loss']:.4f} | Dialogue Turns: {turns}")

    # Evaluate at 50 turns and 100 turns
    bench.evaluate(model, num_turns=50, num_trials=10)
    bench.evaluate(model, num_turns=100, num_trials=10)

if __name__ == "__main__":
    main()
