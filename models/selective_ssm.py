import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .parallel_scan import chunked_parallel_ssm_scan

class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (SSM) layer.
    Features:
    - Input-dependent Delta, B, C parameters (Selective mechanism)
    - O(1) recurrent step for constant-memory inference
    - Chunked Parallel Scan (State-Space Duality) for fast sequence training
    - Continuous-to-discrete zero-order hold (ZOH) discretization
    """
    def __init__(self, d_model: int, d_state: int = 16, d_inner: int = None, dt_rank: int = None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_inner or d_model
        self.dt_rank = dt_rank or max(1, math.ceil(self.d_model / 16))

        # Project input to inner dimension
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        # Delta projection (input-dependent time step)
        self.dt_proj_rank = nn.Linear(self.d_inner, self.dt_rank, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Input-dependent B and C
        self.b_proj = nn.Linear(self.d_inner, self.d_state, bias=False)
        self.c_proj = nn.Linear(self.d_inner, self.d_state, bias=False)

        # Log A matrix (diagonal, learnable initialization)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # D skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward_recurrent_step(self, x: torch.Tensor, prev_state: torch.Tensor):
        """
        Step-by-step O(1) inference without KV cache.
        x: (batch_size, d_model)
        prev_state: (batch_size, d_inner, d_state)
        Returns:
            y: (batch_size, d_model)
            next_state: (batch_size, d_inner, d_state)
        """
        u, gate = self.in_proj(x).chunk(2, dim=-1) # (B, d_inner) each

        # Discretization
        dt = F.softplus(self.dt_proj(self.dt_proj_rank(u))) # (B, d_inner)
        B_mat = self.b_proj(u) # (B, d_state)
        C_mat = self.c_proj(u) # (B, d_state)

        A = -torch.exp(self.A_log) # (d_inner, d_state)
        # dA = exp(dt * A) -> (B, d_inner, d_state)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
        # dB = dt * B -> (B, d_inner, d_state)
        dB = dt.unsqueeze(-1) * B_mat.unsqueeze(1)

        # State transition: s_t = dA * s_{t-1} + dB * u_t
        next_state = dA * prev_state + dB * u.unsqueeze(-1) # (B, d_inner, d_state)

        # Output: y = C * s_t + D * u
        y_ssm = torch.sum(next_state * C_mat.unsqueeze(1), dim=-1) # (B, d_inner)
        y = (y_ssm + self.D * u) * F.silu(gate)
        out = self.out_proj(y)
        return out, next_state

    def forward_parallel(self, x: torch.Tensor, chunk_size: int = 32):
        """
        Chunked parallel forward pass for training.
        x: (batch_size, seq_len, d_model)
        """
        B, L, D = x.shape
        u_all, gate_all = self.in_proj(x).chunk(2, dim=-1) # (B, L, d_inner)

        dt_all = F.softplus(self.dt_proj(self.dt_proj_rank(u_all))) # (B, L, d_inner)
        B_all = self.b_proj(u_all) # (B, L, d_state)
        C_all = self.c_proj(u_all) # (B, L, d_state)

        A = -torch.exp(self.A_log) # (d_inner, d_state)
        dA_all = torch.exp(dt_all.unsqueeze(-1) * A.view(1, 1, self.d_inner, self.d_state)) # (B, L, d_inner, d_state)
        dB_all = dt_all.unsqueeze(-1) * B_all.unsqueeze(2) # (B, L, d_inner, d_state)

        y_ssm, final_state = chunked_parallel_ssm_scan(
            u=u_all,
            dA=dA_all,
            dB=dB_all,
            C_mat=C_all,
            chunk_size=chunk_size
        )

        y = (y_ssm + self.D * u_all) * F.silu(gate_all)
        out = self.out_proj(y)
        return out, final_state

    def forward(self, x: torch.Tensor, init_state: torch.Tensor = None, use_parallel: bool = True):
        """
        Sequence forward pass. Uses parallel scan when sequence length > 32 and no custom initial state.
        """
        if use_parallel and init_state is None and x.shape[1] >= 16:
            return self.forward_parallel(x)

        B, L, D = x.shape
        if init_state is None:
            state = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        else:
            state = init_state

        outputs = []
        for t in range(L):
            out_t, state = self.forward_recurrent_step(x[:, t, :], state)
            outputs.append(out_t)

        out = torch.stack(outputs, dim=1)
        return out, state
