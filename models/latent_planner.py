import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentPlanner(nn.Module):
    """
    Adaptive Zero-Token Latent Rollout Planner.
    
    Features:
    - Adaptive Rollout Depth: Dynamically scales thought depth K based on surprise/uncertainty.
    - Multi-branch Counterfactual Imagination: Explores parallel hypothesis trajectories in latent space.
    - Value-Guided State Selection: Picks the most consistent imagined target state without decoding tokens.
    """
    def __init__(
        self,
        rssm_cell: nn.Module,
        d_model: int,
        max_rollout_steps: int = 6,
        num_branches: int = 4,
        temperature: float = 1.0,
        evict_scratchpad_on_completion: bool = True
    ):
        super().__init__()
        self.rssm_cell = rssm_cell
        self.d_model = d_model
        self.max_rollout_steps = max_rollout_steps
        self.num_branches = num_branches
        self.temperature = temperature
        self.evict_scratchpad_on_completion = evict_scratchpad_on_completion

        # Value Head: Estimates future task reward / state consistency
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

        # Latent thought perturbation generator for branch exploration
        self.thought_perturb = nn.Linear(d_model, d_model)

    def compute_adaptive_depth(self, surprise: torch.Tensor):
        """
        Determines rollout depth K dynamically:
        K = clamp(ceil(surprise * max_rollout_steps), min=1, max=max_rollout_steps)
        """
        depths = torch.clamp(
            torch.ceil(surprise * self.max_rollout_steps).long(),
            min=1,
            max=self.max_rollout_steps
        )
        return depths.max().item() # Max depth required across the batch

    def forward(self, curr_h: torch.Tensor, curr_ssm_state: torch.Tensor, surprise: torch.Tensor = None):
        """
        Executes adaptive parallel latent rollouts.
        curr_h: (B, d_model)
        curr_ssm_state: (B, d_inner, d_state)
        surprise: (B, 1) optional surprise metric from previous step
        """
        B = curr_h.shape[0]
        M = self.num_branches

        if surprise is not None:
            K = self.compute_adaptive_depth(surprise)
        else:
            K = self.max_rollout_steps

        # Replicate states across M branches: (B*M, ...)
        h = curr_h.unsqueeze(1).repeat(1, M, 1).view(B * M, self.d_model)
        
        d_inner = curr_ssm_state.shape[1]
        d_state = curr_ssm_state.shape[2]
        ssm_state = curr_ssm_state.unsqueeze(1).repeat(1, M, 1, 1).view(B * M, d_inner, d_state)

        # Inject stochastic exploration across branches
        noise = torch.randn_like(h) * 0.08
        h = h + self.thought_perturb(noise)

        # Execute K-step internal rollout in latent space
        for step_idx in range(K):
            h, ssm_state, _ = self.rssm_cell.step(
                x_t=None,
                prev_h=h,
                prev_ssm_state=ssm_state,
                use_posterior=False,
                temperature=self.temperature
            )

        # Score imagined endpoints
        scores = self.value_head(h).view(B, M)
        best_branch_idx = torch.argmax(scores, dim=-1) # (B,)

        h_reshaped = h.view(B, M, self.d_model)
        ssm_reshaped = ssm_state.view(B, M, d_inner, d_state)

        batch_indices = torch.arange(B, device=curr_h.device)
        best_h = h_reshaped[batch_indices, best_branch_idx]
        best_ssm_state = ssm_reshaped[batch_indices, best_branch_idx]

        if self.evict_scratchpad_on_completion and hasattr(self.rssm_cell, "evict_scratchpad"):
            best_ssm_state = self.rssm_cell.evict_scratchpad(best_ssm_state)

        return best_h, best_ssm_state, scores, K
