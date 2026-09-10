import torch
import torch.nn as nn
import torch.nn.functional as F
from .cognitive_forgetting_ssm import CognitiveForgettingSSM
from .selective_ssm import SelectiveSSM

class RSSMCell(nn.Module):
    """
    Hierarchical RSSM Cell with Discrete Categorical Latents (DreamerV3 style)
    and Cognitive Active Forgetting Engine (CAFE).
    
    Features:
    1. Categorical Latents: N_cat categorical distributions of K_classes each.
       - Straight-Through Gumbel-Softmax estimator for gradient flow without posterior collapse.
       - Exact analytical discrete KL divergence.
    2. Deterministic Memory: Cognitive Active Forgetting SSM (CAFE) state with
       multi-scale channel lifetimes (persistent, contextual, ephemeral scratchpad).
    3. Surprise Gating: KL(Posterior || Prior) dynamically regulates storage and protects memory.
    4. Ephemeral Scratchpad Eviction: Instant zeroing of scratchpad registers upon thought conclusion.
    5. Semantic Subspace Nullification: Erases obsolete variable states via orthogonal projection.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        num_categoricals: int = 16,
        num_classes: int = 16,
        adaptive_threshold: float = 0.5,
        use_cognitive_forgetting: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.num_categoricals = num_categoricals
        self.num_classes = num_classes
        self.latent_dim = num_categoricals * num_classes
        self.adaptive_threshold = adaptive_threshold
        self.use_cognitive_forgetting = use_cognitive_forgetting

        # SSM for deterministic state transition with Active Forgetting
        if use_cognitive_forgetting:
            self.ssm = CognitiveForgettingSSM(d_model=d_model, d_state=d_state)
        else:
            self.ssm = SelectiveSSM(d_model=d_model, d_state=d_state)

        # Input fusion projection: projects [x_t, flattened_categorical_z] to d_model
        self.fuse_proj = nn.Linear(d_model + self.latent_dim, d_model)

        # Prior network: p(z_t | h_{t-1}) -> logits of shape (B, N_cat, K_cls)
        self.prior_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, self.latent_dim)
        )

        # Posterior network: q(z_t | h_{t-1}, x_t) -> logits of shape (B, N_cat, K_cls)
        self.post_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, self.latent_dim)
        )

        # Surprise dynamic gate: modulates SSM update rate
        self.surprise_gate = nn.Sequential(
            nn.Linear(1, d_model),
            nn.Sigmoid()
        )

    def compute_categorical_kl(self, post_logits, prior_logits):
        """
        Exact analytical KL divergence between categorical distributions:
        KL(q || p) = sum_i sum_k q_{i,k} * (log q_{i,k} - log p_{i,k})
        """
        post_log_probs = F.log_softmax(post_logits, dim=-1)
        post_probs = torch.exp(post_log_probs)
        prior_log_probs = F.log_softmax(prior_logits, dim=-1)

        # Sum over classes (dim=-1) and average/sum over categorical variables (dim=-2)
        kl = torch.sum(post_probs * (post_log_probs - prior_log_probs), dim=-1) # (B, N_cat)
        total_kl = torch.mean(kl, dim=-1, keepdim=True) # (B, 1)
        return total_kl

    def sample_categorical(self, logits, temperature: float = 1.0, hard: bool = True):
        """
        Samples categorical latent using Gumbel-Softmax with Straight-Through (ST) estimator.
        """
        if self.training:
            z_sample = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)
        else:
            # Greedy / Mode evaluation
            argmax_idx = torch.argmax(logits, dim=-1)
            z_sample = F.one_hot(argmax_idx, num_classes=self.num_classes).float()
        return z_sample

    def step(
        self,
        x_t: torch.Tensor,
        prev_h: torch.Tensor,
        prev_ssm_state: torch.Tensor,
        use_posterior: bool = True,
        temperature: float = 1.0,
        is_boundary: torch.Tensor = None,
        scratchpad_flush: bool = False
    ):
        """
        Single recurrent step execution with O(1) memory and active cognitive forgetting.
        """
        B = prev_h.shape[0]

        # 1. Prior distribution: p(z_t | prev_h)
        prior_logits_flat = self.prior_net(prev_h)
        prior_logits = prior_logits_flat.view(B, self.num_categoricals, self.num_classes)

        # 2. Posterior distribution: q(z_t | prev_h, x_t)
        if use_posterior and x_t is not None:
            post_logits_flat = self.post_net(torch.cat([prev_h, x_t], dim=-1))
            post_logits = post_logits_flat.view(B, self.num_categoricals, self.num_classes)
            z_sample = self.sample_categorical(post_logits, temperature=temperature, hard=True)
            kl_div = self.compute_categorical_kl(post_logits, prior_logits)
        else:
            # Prior generation / Latent rollout mode
            post_logits = prior_logits
            z_sample = self.sample_categorical(prior_logits, temperature=temperature, hard=True)
            kl_div = torch.zeros(B, 1, device=prev_h.device)

        # Flatten categorical sample for state integration: (B, N_cat * K_cls)
        z_flat = z_sample.view(B, self.latent_dim)

        # 3. Surprise Metric & Dynamic Gating
        surprise = F.softplus(kl_div)
        gate = self.surprise_gate(surprise)

        # 4. Deterministic Transition via Cognitive SSM
        if x_t is not None:
            fused_in = self.fuse_proj(torch.cat([x_t, z_flat], dim=-1))
        else:
            # Pure latent imagination step
            fused_in = self.fuse_proj(torch.cat([prev_h, z_flat], dim=-1))

        if self.use_cognitive_forgetting:
            next_h, next_ssm_state, ssm_info = self.ssm.forward_recurrent_step(
                fused_in,
                prev_ssm_state,
                surprise=surprise,
                is_boundary=is_boundary,
                scratchpad_flush=scratchpad_flush
            )
        else:
            fused_in = fused_in * (1.0 + gate)
            next_h, next_ssm_state = self.ssm.forward_recurrent_step(fused_in, prev_ssm_state)
            ssm_info = {}

        # 5. Adaptive Thought Trigger Signal
        trigger_thought = (surprise > self.adaptive_threshold).squeeze(-1) # (B,)

        step_info = {
            "prior_logits": prior_logits,
            "post_logits": post_logits,
            "kl_div": kl_div,
            "surprise": surprise,
            "z_sample": z_sample,
            "trigger_thought": trigger_thought,
            "ssm_info": ssm_info
        }
        return next_h, next_ssm_state, step_info

    def nullify_subspace(self, ssm_state: torch.Tensor, obsolete_vector: torch.Tensor):
        """
        Projects SSM state to orthogonal complement of obsolete entity direction.
        """
        if hasattr(self.ssm, "nullify_subspace"):
            return self.ssm.nullify_subspace(ssm_state, obsolete_vector)
        return ssm_state

    def evict_scratchpad(self, ssm_state: torch.Tensor):
        """
        Immediately zeroes out ephemeral reasoning scratchpad channels.
        """
        if hasattr(self.ssm, "n_scratchpad") and self.ssm.n_scratchpad > 0:
            cleansed = ssm_state.clone()
            cleansed[:, -self.ssm.n_scratchpad:, :] = 0.0
            return cleansed
        return ssm_state
