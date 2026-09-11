import torch
import torch.nn as nn
import torch.nn.functional as F

class AnchoredLatentPlanner(nn.Module):
    """
    Mitigation Mechanism 1 & 2:
    - Anchored Latent Rollout: Regulates semantic drift at deep horizons (K >= 15)
      by soft-anchoring latent thought trajectories to context anchor h_0.
    - On-Demand Thought Prober: Decodes intermediate latent mental states h_{t+k}
      into human-auditable text tokens/lemmas on demand without slowing normal inference.
    """
    def __init__(self, rssm_cell, d_model: int, vocab_size: int, max_rollout_steps: int = 25, num_branches: int = 4):
        super().__init__()
        self.rssm_cell = rssm_cell
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_rollout_steps = max_rollout_steps
        self.num_branches = num_branches

        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

        # On-Demand Thought Probing Head: Maps latent state to vocabulary distribution
        self.thought_prober = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, vocab_size)
        )

    def forward(self, h_t: torch.Tensor, ssm_state: torch.Tensor, depth_k: int = 4, use_anchor: bool = True):
        """
        Executes Anchored Latent Rollout up to depth_k.
        """
        B = h_t.shape[0]
        M = self.num_branches
        h_0 = h_t.clone() # Context Anchor

        # Expand for M parallel thought branches
        h_branches = h_t.unsqueeze(1).repeat(1, M, 1).view(B * M, self.d_model)
        ssm_branches = ssm_state.unsqueeze(1).repeat(1, M, 1, 1).view(B * M, self.d_model, self.rssm_cell.d_state)
        h_0_exp = h_0.unsqueeze(1).repeat(1, M, 1).view(B * M, self.d_model)

        all_step_states = []

        for step in range(depth_k):
            # Unroll recurrent world model
            next_h, next_ssm, _ = self.rssm_cell.step(
                x_t=None, prev_h=h_branches, prev_ssm_state=ssm_branches, use_posterior=False
            )
            
            # Mechanism 1: Soft Context Anchoring
            if use_anchor and step > 0:
                cos_sim = F.cosine_similarity(next_h, h_0_exp, dim=-1).unsqueeze(-1)
                alpha = torch.sigmoid(cos_sim * 2.0)
                next_h = alpha * next_h + (1.0 - alpha) * h_0_exp

            # Add stochastic exploration noise across branches
            if M > 1:
                noise = torch.randn_like(next_h) * 0.05
                next_h = next_h + noise

            h_branches = next_h
            ssm_branches = next_ssm
            all_step_states.append(h_branches.view(B, M, self.d_model))

        # Evaluate candidate thought trajectories with Value Head
        val_scores = self.value_head(h_branches).view(B, M) # (B, M)
        best_branch_idx = torch.argmax(val_scores, dim=-1) # (B,)

        # Gather winning thoughts
        batch_idx = torch.arange(B, device=h_t.device)
        best_h = h_branches.view(B, M, self.d_model)[batch_idx, best_branch_idx]
        best_ssm = ssm_branches.view(B, M, self.d_model, self.rssm_cell.d_state)[batch_idx, best_branch_idx]

        return best_h, best_ssm, val_scores, all_step_states

    def probe_thought_text(self, latent_states: list, tokenizer):
        """
        Mechanism 2: On-Demand Verbalization of Latent Thoughts.
        Decodes a list of latent step tensors into human-readable text strings.
        """
        decoded_steps = []
        for step_idx, step_tensor in enumerate(latent_states):
            # step_tensor: (B, M, d_model) -> pick branch 0 or best branch
            h_sample = step_tensor[0, 0] # (d_model,)
            logits = self.thought_prober(h_sample.unsqueeze(0)) # (1, vocab_size)
            top_toks = torch.topk(logits, k=3, dim=-1).indices[0].tolist()
            decoded_words = [tokenizer.decode([tok]).strip() for tok in top_toks]
            decoded_steps.append(f"Step {step_idx+1}: [{' | '.join(decoded_words)}]")
        return decoded_steps
