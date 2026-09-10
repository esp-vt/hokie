import unittest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.selective_ssm import SelectiveSSM
from models.rssm_cell import RSSMCell
from models.latent_planner import LatentPlanner
from models.neuroworld import NeuroWorldLM

class TestNeuroWorldArchitecture(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.d_model = 64
        self.d_state = 16
        self.num_cat = 4
        self.num_cls = 4
        self.vocab_size = 128

    def test_01_selective_ssm_parallel_equivalence(self):
        """Tests that Chunked Parallel Scan matches Sequential Step within float32 precision."""
        ssm = SelectiveSSM(d_model=self.d_model, d_state=self.d_state).to(self.device)
        x = torch.randn(2, 64, self.d_model, device=self.device)
        
        out_seq, s_seq = ssm.forward(x, use_parallel=False)
        out_par, s_par = ssm.forward(x, use_parallel=True)
        
        max_diff = torch.max(torch.abs(out_seq - out_par)).item()
        self.assertLess(max_diff, 1e-4, f"Parallel scan diverged from sequential: {max_diff}")

    def test_02_rssm_categorical_gumbel_st(self):
        """Tests that Categorical Latents compute valid analytical KL and produce gradients."""
        cell = RSSMCell(
            d_model=self.d_model,
            d_state=self.d_state,
            num_categoricals=self.num_cat,
            num_classes=self.num_cls
        ).to(self.device)

        prev_h = torch.randn(2, self.d_model, requires_grad=True, device=self.device)
        prev_ssm = torch.zeros(2, self.d_model, self.d_state, device=self.device)
        x_t = torch.randn(2, self.d_model, device=self.device)

        next_h, next_ssm, info = cell.step(x_t, prev_h, prev_ssm, use_posterior=True)
        
        # Test KL divergence is non-negative
        self.assertTrue(torch.all(info["kl_div"] >= -1e-5), "KL divergence must be non-negative")
        
        # Test backward gradient flow
        loss = next_h.sum() + info["kl_div"].sum()
        loss.backward()
        self.assertIsNotNone(prev_h.grad, "Gradients must flow through Gumbel-Softmax ST")

    def test_03_zero_token_latent_planner(self):
        """Tests that LatentPlanner performs multi-branch rollout without tensor shape corruption."""
        cell = RSSMCell(
            d_model=self.d_model,
            d_state=self.d_state,
            num_categoricals=self.num_cat,
            num_classes=self.num_cls
        ).to(self.device)
        
        planner = LatentPlanner(rssm_cell=cell, d_model=self.d_model, max_rollout_steps=4, num_branches=3).to(self.device)
        
        h = torch.randn(2, self.d_model, device=self.device)
        ssm = torch.zeros(2, self.d_model, self.d_state, device=self.device)
        
        best_h, best_ssm, scores, depth = planner(h, ssm)
        self.assertEqual(best_h.shape, (2, self.d_model))
        self.assertEqual(scores.shape, (2, 3))
        self.assertGreaterEqual(depth, 1)

    def test_04_neuroworld_end_to_end_lm(self):
        """Tests complete NeuroWorldLM training step and generation modes."""
        model = NeuroWorldLM(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            d_state=self.d_state,
            num_categoricals=self.num_cat,
            num_classes=self.num_cls,
            num_layers=2
        ).to(self.device)

        input_ids = torch.randint(0, self.vocab_size, (2, 32), device=self.device)
        logits, loss, metrics, _ = model(input_ids, targets=input_ids)
        
        self.assertEqual(logits.shape, (2, 32, self.vocab_size))
        self.assertFalse(torch.isnan(loss), "Loss must not be NaN")
        
        # Test Direct and Latent Thought Generation
        prompt = input_ids[:, :8]
        gen_dir = model.generate(prompt, max_new_tokens=6)
        self.assertEqual(gen_dir.shape, (2, 6))
        
        gen_thought, _, _ = model.generate_with_adaptive_thought(prompt, max_new_tokens=6)
        self.assertEqual(gen_thought.shape, (2, 6))

if __name__ == "__main__":
    print("=" * 70)
    print("  Running NeuroWorld-LM Automated Unit Test Suite  ")
    print("=" * 70)
    unittest.main()
