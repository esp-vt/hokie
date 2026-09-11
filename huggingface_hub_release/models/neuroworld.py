import torch
import torch.nn as nn
import torch.nn.functional as F
from .rssm_cell import RSSMCell
from .latent_planner import LatentPlanner

class NeuroWorldLM(nn.Module):
    """
    NeuroWorld-LM: Scalable World-Model Language Model with Categorical RSSM
    and Adaptive Zero-Token Latent Rollout Planning.
    """
    def __init__(
        self,
        vocab_size: int = 4096,
        d_model: int = 256,
        d_state: int = 16,
        num_categoricals: int = 16,
        num_classes: int = 16,
        num_layers: int = 2,
        max_rollout_steps: int = 6,
        num_branches: int = 4,
        kl_balance_weight: float = 0.8,
        free_bits_tau: float = 0.05
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_state = d_state
        self.num_categoricals = num_categoricals
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.kl_balance_weight = kl_balance_weight
        self.free_bits_tau = free_bits_tau

        # Token Embedding
        self.tok_embed = nn.Embedding(vocab_size, d_model)

        # RSSM Layers with Categorical Latents
        self.layers = nn.ModuleList([
            RSSMCell(
                d_model=d_model,
                d_state=d_state,
                num_categoricals=num_categoricals,
                num_classes=num_classes
            )
            for _ in range(num_layers)
        ])

        # Normalization
        self.ln_f = nn.LayerNorm(d_model)

        # Zero-Token Latent Planner
        self.planner = LatentPlanner(
            rssm_cell=self.layers[-1],
            d_model=d_model,
            max_rollout_steps=max_rollout_steps,
            num_branches=num_branches
        )

        # Language Modeling Head with weight tying
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_embed.weight

    def init_hidden(self, batch_size: int, device: torch.device):
        h_list = [torch.zeros(batch_size, self.d_model, device=device) for _ in range(self.num_layers)]
        ssm_states = [torch.zeros(batch_size, self.d_model, self.d_state, device=device) for _ in range(self.num_layers)]
        return h_list, ssm_states

    def nullify_entity(self, ssm_states: list, entity_vector: torch.Tensor):
        """
        Applies Semantic Subspace Nullification across all model layers.
        Guarantees zero mutual information between obsolete entity and model working memory.
        """
        cleansed_states = []
        for l_idx, layer in enumerate(self.layers):
            cleansed = layer.nullify_subspace(ssm_states[l_idx], entity_vector)
            cleansed_states.append(cleansed)
        return cleansed_states

    def scrub_pii_tokens(self, ssm_states: list, pii_token_ids: torch.Tensor):
        """
        Takes raw discrete token IDs of sensitive PII (e.g. API keys, SSN digits, passwords),
        projects their multi-token embedding subspace, and performs instantaneous orthogonal
        nullification across all layer states.
        
        pii_token_ids: (B, K) or (K,)
        """
        if pii_token_ids.dim() == 1:
            pii_token_ids = pii_token_ids.unsqueeze(0) # (1, K)
        pii_embeds = self.tok_embed(pii_token_ids) # (B, K, d_model)
        return self.nullify_entity(ssm_states, pii_embeds)

    def evict_scratchpads(self, ssm_states: list):
        """
        Instantly clears ephemeral reasoning scratchpad channels across all layers.
        """
        cleansed_states = []
        for l_idx, layer in enumerate(self.layers):
            cleansed = layer.evict_scratchpad(ssm_states[l_idx])
            cleansed_states.append(cleansed)
        return cleansed_states

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor = None,
        use_posterior: bool = True,
        boundary_mask: torch.Tensor = None
    ):
        """
        Full sequence forward pass with Cognitive Active Forgetting (CAFE).
        input_ids: (B, L)
        targets: (B, L)
        boundary_mask: (B, L) binary mask indicating topic/turn change points
        """
        B, L = input_ids.shape
        device = input_ids.device

        x = self.tok_embed(input_ids)
        h_list, ssm_states = self.init_hidden(B, device)

        all_logits = []
        total_kl = 0.0
        surprise_history = []

        for t in range(L):
            curr_x = x[:, t, :]
            is_b = boundary_mask[:, t:t+1] if boundary_mask is not None else None
            for l_idx, layer in enumerate(self.layers):
                curr_x, ssm_states[l_idx], step_info = layer.step(
                    x_t=curr_x,
                    prev_h=h_list[l_idx],
                    prev_ssm_state=ssm_states[l_idx],
                    use_posterior=use_posterior,
                    is_boundary=is_b
                )
                h_list[l_idx] = curr_x
                if l_idx == self.num_layers - 1:
                    kl = step_info["kl_div"]
                    clamped_kl = torch.clamp(kl, min=self.free_bits_tau).mean()
                    total_kl += clamped_kl
                    surprise_history.append(step_info["surprise"].mean().item())

            normed = self.ln_f(curr_x)
            logits_t = self.lm_head(normed)
            all_logits.append(logits_t)

        logits = torch.stack(all_logits, dim=1)

        loss = None
        metrics = {}
        if targets is not None:
            token_loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            avg_kl = total_kl / L
            loss = token_loss + 0.1 * avg_kl

            metrics = {
                "loss": loss.item(),
                "token_loss": token_loss.item(),
                "kl_div": avg_kl.item(),
                "mean_surprise": sum(surprise_history) / len(surprise_history) if surprise_history else 0.0
            }

        return logits, loss, metrics, (h_list, ssm_states)

    @torch.no_grad()
    def generate(self, prompt_tokens: torch.Tensor, max_new_tokens: int = 50, temperature: float = 1.0):
        """
        Standard O(1) Memory KV-Free Autoregressive Generation.
        """
        self.eval()
        B, L = prompt_tokens.shape
        device = prompt_tokens.device

        h_list, ssm_states = self.init_hidden(B, device)

        for t in range(L):
            tok = prompt_tokens[:, t]
            x_t = self.tok_embed(tok)
            for l_idx, layer in enumerate(self.layers):
                x_t, ssm_states[l_idx], _ = layer.step(
                    x_t=x_t,
                    prev_h=h_list[l_idx],
                    prev_ssm_state=ssm_states[l_idx],
                    use_posterior=False
                )
                h_list[l_idx] = x_t

        generated = []
        last_h = h_list[-1]

        for _ in range(max_new_tokens):
            normed = self.ln_f(last_h)
            logits = self.lm_head(normed) / temperature
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            generated.append(next_tok)

            x_next = self.tok_embed(next_tok.squeeze(-1))
            for l_idx, layer in enumerate(self.layers):
                x_next, ssm_states[l_idx], _ = layer.step(
                    x_t=x_next,
                    prev_h=h_list[l_idx],
                    prev_ssm_state=ssm_states[l_idx],
                    use_posterior=False
                )
                h_list[l_idx] = x_next
            last_h = h_list[-1]

        return torch.cat(generated, dim=-1)

    @torch.no_grad()
    def generate_with_adaptive_thought(self, prompt_tokens: torch.Tensor, max_new_tokens: int = 10):
        """
        Adaptive Zero-Token Reasoning Generation.
        Triggers variable-depth Latent Rollouts based on prompt surprise/uncertainty.
        """
        self.eval()
        B, L = prompt_tokens.shape
        device = prompt_tokens.device

        h_list, ssm_states = self.init_hidden(B, device)
        last_surprise = torch.zeros(B, 1, device=device)

        for t in range(L):
            tok = prompt_tokens[:, t]
            x_t = self.tok_embed(tok)
            for l_idx, layer in enumerate(self.layers):
                x_t, ssm_states[l_idx], step_info = layer.step(
                    x_t=x_t,
                    prev_h=h_list[l_idx],
                    prev_ssm_state=ssm_states[l_idx],
                    use_posterior=False
                )
                h_list[l_idx] = x_t
                if l_idx == self.num_layers - 1:
                    last_surprise = step_info["surprise"]

        # Run Adaptive Latent Planner
        best_h, best_ssm, branch_scores, depth_k = self.planner(
            h_list[-1], ssm_states[-1], surprise=last_surprise
        )
        h_list[-1] = best_h
        ssm_states[-1] = best_ssm

        generated = []
        last_h = h_list[-1]
        for _ in range(max_new_tokens):
            normed = self.ln_f(last_h)
            logits = self.lm_head(normed)
            next_tok = torch.argmax(logits, dim=-1, keepdim=True)
            generated.append(next_tok)

            x_next = self.tok_embed(next_tok.squeeze(-1))
            for l_idx, layer in enumerate(self.layers):
                x_next, ssm_states[l_idx], _ = layer.step(
                    x_t=x_next,
                    prev_h=h_list[l_idx],
                    prev_ssm_state=ssm_states[l_idx],
                    use_posterior=False
                )
                h_list[l_idx] = x_next
            last_h = h_list[-1]

        return torch.cat(generated, dim=-1), branch_scores, depth_k
