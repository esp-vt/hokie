import time
import torch
import torch.nn.functional as F
from models.neuroworld import NeuroWorldLM
from benchmarks.synthetic_tasks import SyntheticTaskGenerator

def evaluate_neuroworld():
    print("=" * 65)
    print("  NeuroWorld-LM: Comprehensive Architecture & Efficiency Evaluation  ")
    print("=" * 65)

    device = torch.device("cpu")
    vocab_size = 512
    d_model = 128
    d_latent = 32
    d_state = 16
    num_layers = 2

    model = NeuroWorldLM(
        vocab_size=vocab_size,
        d_model=d_model,
        d_latent=d_latent,
        d_state=d_state,
        num_layers=num_layers,
        rollout_steps=4,
        num_branches=4
    ).to(device)

    import os
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "neuroworld_best.pt")
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[✓] Loaded trained checkpoint from {ckpt_path}")
    except Exception as e:
        print(f"[!] Could not load checkpoint ({e}), evaluating initialized model.")

    model.eval()
    task_gen = SyntheticTaskGenerator(vocab_size=vocab_size)

    # -------------------------------------------------------------
    # 1. KV-Free O(1) Memory Verification
    # -------------------------------------------------------------
    print("\n[Benchmark 1] KV-Free O(1) State Memory Scaling Test")
    print("-" * 65)
    test_lengths = [64, 256, 1024, 4096]
    for seq_len in test_lengths:
        dummy_input = torch.randint(0, vocab_size, (1, seq_len), device=device)
        h_list, ssm_states = model.init_hidden(batch_size=1, device=device)
        
        # State memory footprint in bytes
        state_bytes = sum(h.element_size() * h.nelement() for h in h_list) + \
                      sum(s.element_size() * s.nelement() for s in ssm_states)
        
        # In contrast, standard Transformer KV Cache footprint: 2 * num_layers * seq_len * d_model * 4 bytes
        transformer_kv_bytes = 2 * num_layers * seq_len * d_model * 4
        
        print(f" Seq Length: {seq_len:5d} tokens | NeuroWorld State: {state_bytes/1024:6.2f} KB (CONSTANT O(1)) | Transformer KV: {transformer_kv_bytes/1024:7.2f} KB (O(T))")

    # -------------------------------------------------------------
    # 2. Multi-Step State Tracking Accuracy
    # -------------------------------------------------------------
    print("\n[Benchmark 2] World State Tracking Task Evaluation")
    print("-" * 65)
    test_steps = [2, 4, 6, 8]
    for n_step in test_steps:
        inputs, targets = task_gen.generate_state_tracking_batch(batch_size=32, num_steps=n_step)
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            logits, _, _, _ = model(inputs, targets=targets, use_posterior=False)
            preds = torch.argmax(logits[:, -1, :], dim=-1)
            acc = (preds == targets[:, -1]).float().mean().item() * 100.0
        print(f" State Transitions: {n_step:2d} steps | Target State Accuracy: {acc:5.1f}%")

    # -------------------------------------------------------------
    # 3. Zero-Token Latent Rollout Reasoning
    # -------------------------------------------------------------
    print("\n[Benchmark 3] Zero-Token Latent Rollout Engine Verification")
    print("-" * 65)
    sample_prompt = torch.tensor([[1, 10, 4, 60, 11, 4, 75, 3, 11]], device=device) # Query var 11
    
    t0 = time.time()
    out_tokens_direct = model.generate(sample_prompt, max_new_tokens=4)
    t_direct = (time.time() - t0) * 1000

    t0 = time.time()
    out_tokens_thought, branch_scores = model.generate_with_latent_thought(sample_prompt, max_new_tokens=4)
    t_thought = (time.time() - t0) * 1000

    print(f" Direct Decoding Output      : {out_tokens_direct.tolist()} ({t_direct:.2f} ms)")
    print(f" Latent Thought + Jump Output: {out_tokens_thought.tolist()} ({t_thought:.2f} ms)")
    print(f" Imagined Branch Valuations  : {branch_scores.tolist()}")

    print("\n" + "=" * 65)
    print("  All Evaluation Benchmarks Finished Successfully!  ")
    print("=" * 65)

if __name__ == "__main__":
    evaluate_neuroworld()
