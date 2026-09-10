import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class MQARBenchmark:
    """
    Multi-Query Associative Recall (MQAR) Benchmark Suite.
    Evaluates associative memory retrieval across long horizons (up to 16,384 tokens).
    """
    def __init__(self, vocab_size: int = 1024, num_kv_pairs: int = 16):
        self.vocab_size = vocab_size
        self.num_kv_pairs = num_kv_pairs
        self.bos = 1
        self.eos = 2
        self.query_marker = 3
        self.assign_marker = 4
        self.key_offset = 10
        self.val_offset = 200
        self.noise_offset = 500

    def generate_sample(self, seq_len: int):
        """
        Generates a single MQAR sequence of exact length `seq_len`.
        Format: [BOS] (K_1 = V_1) [Distractors...] (K_n = V_n) [QUERY] K_target -> V_target
        """
        num_pairs = min(self.num_kv_pairs, max(4, seq_len // 128))
        keys = random.sample(range(self.key_offset, self.key_offset + 100), num_pairs)
        values = random.sample(range(self.val_offset, self.val_offset + 100), num_pairs)
        kv_map = dict(zip(keys, values))

        seq = [self.bos]
        
        # Calculate noise distribution
        fixed_tokens = 1 + (num_pairs * 3) + 2 # BOS + (K, =, V)*N + (QUERY, K)
        noise_tokens_total = max(0, seq_len - fixed_tokens)
        noise_per_slot = noise_tokens_total // num_pairs

        for k, v in zip(keys, values):
            seq.extend([k, self.assign_marker, v])
            # Insert distractor noise tokens
            noise_chunk = [random.randint(self.noise_offset, self.vocab_size - 1) for _ in range(noise_per_slot)]
            seq.extend(noise_chunk)

        # Query a random key from the past
        target_k = random.choice(keys)
        target_v = kv_map[target_k]

        seq.extend([self.query_marker, target_k])

        # If any length deficit remains, pad with noise before query
        if len(seq) < seq_len:
            pad = [random.randint(self.noise_offset, self.vocab_size - 1) for _ in range(seq_len - len(seq))]
            seq = seq[:-2] + pad + seq[-2:]

        return torch.tensor(seq[:seq_len], dtype=torch.long), target_v

    def evaluate_model_on_lengths(self, model: nn.Module, lengths=[512, 1024, 2048, 4096, 8192, 16384], num_trials=16):
        """
        Evaluates memory retention and accuracy across multiple sequence lengths.
        """
        model.eval()
        device = next(model.parameters()).device
        results = {}

        print("\n" + "=" * 70)
        print("  Multi-Query Associative Recall (MQAR) Evaluation (8k / 16k Scale)  ")
        print("=" * 70)

        for L in lengths:
            correct = 0
            latencies = []
            
            # Check state memory footprint
            h_list, ssm_states = model.init_hidden(batch_size=1, device=device)
            state_kb = (sum(h.element_size() * h.nelement() for h in h_list) +
                        sum(s.element_size() * s.nelement() for s in ssm_states)) / 1024.0
            
            transformer_kv_kb = (2 * model.num_layers * L * model.d_model * 4) / 1024.0

            for _ in range(num_trials):
                input_seq, target_v = self.generate_sample(L)
                input_tensor = input_seq.unsqueeze(0).to(device)

                t0 = time.time()
                with torch.no_grad():
                    # Recurrent step or sequence forward
                    logits, _, _, _ = model(input_tensor, use_posterior=False)
                    pred_token = torch.argmax(logits[:, -1, :], dim=-1).item()
                t_eval = (time.time() - t0) * 1000
                latencies.append(t_eval)

                if pred_token == target_v:
                    correct += 1

            acc = (correct / num_trials) * 100.0
            avg_lat = sum(latencies) / len(latencies)
            results[L] = {
                "accuracy": acc,
                "latency_ms": avg_lat,
                "state_kb": state_kb,
                "transformer_kv_kb": transformer_kv_kb
            }

            print(f" Length: {L:6d} tokens | Query Acc: {acc:5.1f}% | Latency: {avg_lat:6.2f} ms | NeuroWorld State: {state_kb:5.1f} KB | Transformer KV: {transformer_kv_kb:8.1f} KB")

        return results
