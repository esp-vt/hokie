import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActiveForgettingSSM(nn.Module):
    """
    Active Semantic Forgetting Engine (ASFE) for NeuroWorld-LM.
    Replaces passive exponential decay with surprise-modulated, topic-divergence aware active flushing.
    
    Mathematical Formulation:
      F_t = sigmoid(W_f * x_t + U_f * h_{t-1} - beta * gamma_t + delta_topic)
      A_t^{active} = exp( - dt * A * (1 + omega * F_t) )
    """
    def __init__(self, d_model: int, d_state: int = 16, d_inner: int = None, dt_rank: int = None, flush_omega: float = 10.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_inner if d_inner is not None else 2 * d_model
        self.dt_rank = dt_rank if dt_rank is not None else max(16, d_model // 16)
        self.flush_omega = flush_omega

        # In / Out projections
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Delta & B, C projections
        self.dt_proj_rank = nn.Linear(self.d_inner, self.dt_rank, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.b_proj = nn.Linear(self.d_inner, self.d_state, bias=False)
        self.c_proj = nn.Linear(self.d_inner, self.d_state, bias=False)

        # Log A matrix
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Active Forgetting Gate Projectors
        self.w_forget = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.w_topic = nn.Linear(self.d_inner, 1, bias=False)
        self.beta_surprise = nn.Parameter(torch.tensor(2.0))

    def forward_step(self, x: torch.Tensor, prev_state: torch.Tensor, surprise: torch.Tensor = None):
        """
        x: (B, d_model)
        prev_state: (B, d_inner, d_state)
        surprise: (B, 1) or scalar KL surprise gamma_t
        """
        B = x.shape[0]
        u, gate = self.in_proj(x).chunk(2, dim=-1) # (B, d_inner)

        # 1. Compute Base Discretization Step
        dt = F.softplus(self.dt_proj(self.dt_proj_rank(u))) # (B, d_inner)
        B_mat = self.b_proj(u) # (B, d_state)
        C_mat = self.c_proj(u) # (B, d_state)
        A = -torch.exp(self.A_log) # (d_inner, d_state)

        # 2. Compute Active Semantic Forget Gate (F_t in [0, 1])
        f_raw = self.w_forget(u)
        topic_shift = self.w_topic(u) # (B, 1)

        if surprise is not None:
            if isinstance(surprise, float):
                surprise = torch.tensor([[surprise]], device=x.device).expand(B, 1)
            f_gate = torch.sigmoid(f_raw - self.beta_surprise * surprise + topic_shift)
        else:
            f_gate = torch.sigmoid(f_raw + topic_shift)

        # 3. Active Hurwitz Decay Modulation
        # F_t -> 1 => Decay amplified by (1 + omega) => Instantaneous Flush
        # F_t -> 0 => Decay suppressed => Permanent Lock
        effective_decay = dt * (1.0 + self.flush_omega * f_gate)
        dA = torch.exp(effective_decay.unsqueeze(-1) * A.unsqueeze(0))
        dB = dt.unsqueeze(-1) * B_mat.unsqueeze(1)

        # 4. State Update
        next_state = dA * prev_state + dB * u.unsqueeze(-1)

        # 5. Output Projection
        y_ssm = torch.sum(next_state * C_mat.unsqueeze(1), dim=-1)
        y = (y_ssm + self.D * u) * F.silu(gate)
        out = self.out_proj(y)

        return out, next_state, f_gate.mean(dim=-1, keepdim=True)
