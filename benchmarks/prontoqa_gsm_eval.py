import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReasoningProtocolBenchmark:
    """
    Reasoning & Deduction Benchmark Suite (PrOntoQA & Symbolic GSM Protocol).
    Tests multi-hop logical deduction and state evolution comparing
    Direct Decoding vs Zero-Token Latent Rollout Planning.
    """
    def __init__(self, vocab_size: int = 1024):
        self.vocab_size = vocab_size
        self.bos = 1
        self.eos = 2
        self.query_tok = 3
        self.is_tok = 4
        self.implies_tok = 5
        self.true_tok = 6
        self.false_tok = 7
        self.entity_offset = 50
        self.concept_offset = 150
        self.number_offset = 300

    def generate_prontoqa_sample(self, num_hops: int = 3):
        """
        Generates a multi-hop deductive ontology:
        Concepts: C0 -> C1 -> C2 -> ... -> C_k
        Rule: C_i implies C_{i+1}
        Fact: Entity E0 is C0
        Query: Is E0 C_k? (True) or Is E0 C_distractor? (False)
        """
        concepts = [self.concept_offset + i for i in range(num_hops + 1)]
        entity = self.entity_offset + random.randint(0, 20)
        distractor_concept = self.concept_offset + 50 + random.randint(0, 20)

        seq = [self.bos]
        # Emit deduction chain rules
        for i in range(num_hops):
            seq.extend([concepts[i], self.implies_tok, concepts[i+1]])

        # Emit ground fact
        seq.extend([entity, self.is_tok, concepts[0]])

        # Target query
        is_true = random.random() > 0.5
        target_concept = concepts[-1] if is_true else distractor_concept
        expected_ans = self.true_tok if is_true else self.false_tok

        seq.extend([self.query_tok, entity, self.is_tok, target_concept])
        return torch.tensor(seq, dtype=torch.long), expected_ans

    def generate_symbolic_gsm_sample(self, num_steps: int = 4):
        """
        Generates multi-step arithmetic / resource evolution problem:
        Variable: V
        Transitions: V = V0; V = V + a1; V = V - a2; ...
        Target: Final value of V
        """
        var = self.entity_offset + random.randint(0, 10)
        curr_val = random.randint(10, 50)
        seq = [self.bos, var, self.is_tok, self.number_offset + curr_val]

        for _ in range(num_steps):
            delta = random.randint(-5, 10)
            curr_val = max(1, curr_val + delta)
            seq.extend([var, self.implies_tok, self.number_offset + curr_val])

        expected_tok = self.number_offset + curr_val
        seq.extend([self.query_tok, var])
        return torch.tensor(seq, dtype=torch.long), expected_tok

    def evaluate_reasoning_suite(self, model: nn.Module, num_samples: int = 32):
        """
        Comprehensive comparison of Direct Decoding vs Zero-Token Latent Rollout.
        """
        model.eval()
        device = next(model.parameters()).device

        print("\n" + "=" * 70)
        print("  PrOntoQA (Multi-Hop Logic) & Symbolic GSM (Arithmetic Dynamics)  ")
        print("=" * 70)

        for task_name, generator, hops_list in [
            ("PrOntoQA (Multi-Hop Deduction)", self.generate_prontoqa_sample, [2, 3, 4, 5]),
            ("Symbolic GSM (State Evolution)", self.generate_symbolic_gsm_sample, [2, 3, 4, 5])
        ]:
            print(f"\n--- Benchmark: {task_name} ---")
            for hops in hops_list:
                direct_correct = 0
                thought_correct = 0
                direct_times = []
                thought_times = []
                thought_depths = []

                for _ in range(num_samples):
                    seq, target = generator(hops)
                    prompt = seq.unsqueeze(0).to(device)

                    # 1. Direct Decoding
                    t0 = time.time()
                    with torch.no_grad():
                        logits, _, _, _ = model(prompt, use_posterior=False)
                        pred_direct = torch.argmax(logits[:, -1, :], dim=-1).item()
                    t_direct = (time.time() - t0) * 1000
                    direct_times.append(t_direct)
                    if pred_direct == target:
                        direct_correct += 1

                    # 2. Zero-Token Latent Rollout Thought Engine
                    t0 = time.time()
                    with torch.no_grad():
                        out_toks, _, depth_k = model.generate_with_adaptive_thought(prompt, max_new_tokens=1)
                        pred_thought = out_toks[0, 0].item()
                    t_thought = (time.time() - t0) * 1000
                    thought_times.append(t_thought)
                    thought_depths.append(depth_k)
                    if pred_thought == target:
                        thought_correct += 1

                acc_dir = (direct_correct / num_samples) * 100.0
                acc_tht = (thought_correct / num_samples) * 100.0
                lat_dir = sum(direct_times) / len(direct_times)
                lat_tht = sum(thought_times) / len(thought_times)
                avg_k = sum(thought_depths) / len(thought_depths)

                print(
                    f" Complexity: {hops} hops | "
                    f"Direct Acc: {acc_dir:5.1f}% ({lat_dir:5.2f}ms) | "
                    f"Latent Thought Acc: {acc_tht:5.1f}% ({lat_tht:5.2f}ms, Depth K={avg_k:.1f}) | "
                    f"Accuracy Delta: {acc_tht - acc_dir:+5.1f}%"
                )
