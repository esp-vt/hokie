import torch
import random

class SyntheticTaskGenerator:
    """
    Generates synthetic benchmarks for:
    1. Associative Recall (Key-Value storage without KV cache)
    2. Multi-Step State Tracking (World model state evolution)
    3. Algorithmic Reasoning Tasks
    """
    def __init__(self, vocab_size: int = 1000, pad_token: int = 0):
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        # Reserve token IDs for special roles
        self.bos_token = 1
        self.eos_token = 2
        self.query_token = 3
        self.assign_token = 4
        self.data_token_offset = 10

    def generate_associative_recall_batch(self, batch_size: int = 16, num_pairs: int = 8, noise_len: int = 64):
        """
        Sequence structure:
        [BOS] (K1, V1) ... [Noise tokens] ... (Kn, Vn) ... [QUERY] K_target -> Target: V_target
        """
        inputs = []
        targets = []

        for _ in range(batch_size):
            keys = random.sample(range(self.data_token_offset, self.data_token_offset + 100), num_pairs)
            values = random.sample(range(self.data_token_offset + 100, self.data_token_offset + 200), num_pairs)
            kv_dict = dict(zip(keys, values))

            seq = [self.bos_token]
            for k, v in zip(keys, values):
                seq.extend([k, self.assign_token, v])
                # Insert random filler noise tokens to test long-range retention
                filler_count = random.randint(2, max(3, noise_len // num_pairs))
                seq.extend([random.randint(self.data_token_offset + 200, self.vocab_size - 1) for _ in range(filler_count)])

            # Query a random key that appeared earlier
            query_k = random.choice(keys)
            expected_v = kv_dict[query_k]

            seq.extend([self.query_token, query_k])
            
            # Autoregressive target is shifted sequence
            input_seq = torch.tensor(seq, dtype=torch.long)
            target_seq = torch.tensor(seq[1:] + [expected_v], dtype=torch.long)

            inputs.append(input_seq)
            targets.append(target_seq)

        # Pad sequences to max length in batch
        max_len = max(x.size(0) for x in inputs)
        padded_inputs = torch.full((batch_size, max_len), self.pad_token, dtype=torch.long)
        padded_targets = torch.full((batch_size, max_len), self.pad_token, dtype=torch.long)

        for i in range(batch_size):
            padded_inputs[i, :inputs[i].size(0)] = inputs[i]
            padded_targets[i, :targets[i].size(0)] = targets[i]

        return padded_inputs, padded_targets

    def generate_state_tracking_batch(self, batch_size: int = 16, num_steps: int = 5):
        """
        Simulates state transitions of multiple variables:
        A = 3; B = 5; A = A + 1; B = A; Query: B?
        """
        inputs = []
        targets = []

        num_vars = 4
        var_tokens = [self.data_token_offset + i for i in range(num_vars)]
        val_offset = self.data_token_offset + 50

        for _ in range(batch_size):
            env_state = {v: 0 for v in var_tokens}
            seq = [self.bos_token]

            for _ in range(num_steps):
                v_target = random.choice(var_tokens)
                new_val = random.randint(1, 20)
                env_state[v_target] = new_val
                seq.extend([v_target, self.assign_token, val_offset + new_val])

            # Query one variable's final state
            q_var = random.choice(var_tokens)
            final_val = env_state[q_var]
            expected_token = val_offset + final_val

            seq.extend([self.query_token, q_var])

            input_seq = torch.tensor(seq, dtype=torch.long)
            target_seq = torch.tensor(seq[1:] + [expected_token], dtype=torch.long)

            inputs.append(input_seq)
            targets.append(target_seq)

        max_len = max(x.size(0) for x in inputs)
        padded_inputs = torch.full((batch_size, max_len), self.pad_token, dtype=torch.long)
        padded_targets = torch.full((batch_size, max_len), self.pad_token, dtype=torch.long)

        for i in range(batch_size):
            padded_inputs[i, :inputs[i].size(0)] = inputs[i]
            padded_targets[i, :targets[i].size(0)] = targets[i]

        return padded_inputs, padded_targets
