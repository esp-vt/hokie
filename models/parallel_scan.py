import torch
import torch.nn as nn
import torch.nn.functional as F

def chunked_parallel_ssm_scan(
    u: torch.Tensor,
    dA: torch.Tensor,
    dB: torch.Tensor,
    C_mat: torch.Tensor,
    chunk_size: int = 32
):
    """
    Chunked Parallel Scan for Selective SSM (State-Space Duality / SSD).
    Reduces sequential time complexity from O(L) to O(L / chunk_size + chunk_size)
    using vectorized matrix operations.
    
    Args:
        u: (B, L, d_inner)
        dA: (B, L, d_inner, d_state) transition decay matrix [exp(dt * A)]
        dB: (B, L, d_inner, d_state) input modulation matrix [dt * B]
        C_mat: (B, L, d_state) output observation matrix
        chunk_size: chunk length for intra-chunk tensor contraction
        
    Returns:
        y_ssm: (B, L, d_inner)
        final_state: (B, d_inner, d_state)
    """
    B, L, D_inner, D_state = dA.shape

    # Handle sequence padding to be a multiple of chunk_size if necessary
    pad_len = (chunk_size - (L % chunk_size)) % chunk_size
    if pad_len > 0:
        u = F.pad(u, (0, 0, 0, pad_len))
        dA = F.pad(dA, (0, 0, 0, 0, 0, pad_len), value=1.0)
        dB = F.pad(dB, (0, 0, 0, 0, 0, pad_len), value=0.0)
        C_mat = F.pad(C_mat, (0, 0, 0, pad_len))

    L_padded = u.shape[1]
    num_chunks = L_padded // chunk_size

    # Reshape into chunks: (B, num_chunks, chunk_size, ...)
    u_chunked = u.view(B, num_chunks, chunk_size, D_inner)
    dA_chunked = dA.view(B, num_chunks, chunk_size, D_inner, D_state)
    dB_chunked = dB.view(B, num_chunks, chunk_size, D_inner, D_state)
    C_chunked = C_mat.view(B, num_chunks, chunk_size, D_state)

    # Input driving term per step: dBu = dB * u -> (B, num_chunks, chunk_size, D_inner, D_state)
    dBu = dB_chunked * u_chunked.unsqueeze(-1)

    # Log-domain cumulative transition across each chunk for numerical stability
    log_dA = torch.log(torch.clamp(dA_chunked, min=1e-12))
    cum_log_dA = torch.cumsum(log_dA, dim=2)
    cum_dA = torch.exp(cum_log_dA)

    # Inter-chunk total decay factor: (B, num_chunks, D_inner, D_state)
    chunk_total_dA = cum_dA[:, :, -1, :, :]

    # State contribution of step j at end of chunk: dBu[:, :, j] * exp(cum_log_dA[-1] - cum_log_dA[j])
    decay_to_end = torch.exp(cum_log_dA[:, :, -1:, :, :] - cum_log_dA)
    chunk_accum_inputs = torch.sum(dBu * decay_to_end, dim=2)

    # Associative scan across chunks
    chunk_states = []
    curr_state = torch.zeros(B, D_inner, D_state, device=u.device, dtype=u.dtype)

    for chunk_idx in range(num_chunks):
        chunk_states.append(curr_state)
        curr_state = chunk_total_dA[:, chunk_idx] * curr_state + chunk_accum_inputs[:, chunk_idx]

    init_chunk_states = torch.stack(chunk_states, dim=1).unsqueeze(2)

    # State component from initial chunk state:
    state_from_init = cum_dA * init_chunk_states

    # Intra-chunk accumulation:
    dBu_expanded = dBu.unsqueeze(2)
    log_dA_i = cum_log_dA.unsqueeze(3)
    log_dA_j = cum_log_dA.unsqueeze(2)

    causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=u.device, dtype=torch.bool))
    causal_mask = causal_mask.view(1, 1, chunk_size, chunk_size, 1, 1)

    decay_matrix = torch.exp(torch.clamp(log_dA_i - log_dA_j, max=0.0))
    decay_matrix = torch.where(causal_mask, decay_matrix, torch.zeros_like(decay_matrix))

    state_from_intra = torch.sum(dBu_expanded * decay_matrix, dim=3)

    full_states = state_from_init + state_from_intra

    # Output projection: y_ssm = sum_state(full_states * C_mat)
    y_chunked = torch.sum(full_states * C_chunked.unsqueeze(3), dim=-1)
    y_ssm = y_chunked.view(B, L_padded, D_inner)

    if pad_len > 0:
        y_ssm = y_ssm[:, :L, :]

    return y_ssm, curr_state
