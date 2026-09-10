import math
import torch
import triton
import triton.language as tl

@triton.jit
def _fused_chunk_scan_kernel(
    X_ptr,          # (B, L, D) Input tensor
    A_ptr,          # (B, L, D) Log-decay factor tensor
    B_ptr,          # (B, L, D) State projection
    Out_ptr,        # (B, L, D) Output tensor
    Final_S_ptr,    # (B, D, N) Final SSM state
    stride_xb, stride_xl, stride_xd,
    stride_ab, stride_al, stride_ad,
    stride_bb, stride_bl, stride_bd,
    stride_ob, stride_ol, stride_od,
    B_dim, L_dim, D_dim,
    BLOCK_D: tl.constexpr,
    CHUNK_SIZE: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D_dim

    # Running recurrent state for this dimension block in fast H100 SRAM
    running_state = tl.zeros([BLOCK_D], dtype=tl.float32)

    for l_idx in range(0, L_dim):
        x_ptrs = X_ptr + pid_b * stride_xb + l_idx * stride_xl + offs_d * stride_xd
        a_ptrs = A_ptr + pid_b * stride_ab + l_idx * stride_al + offs_d * stride_ad
        b_ptrs = B_ptr + pid_b * stride_bb + l_idx * stride_bl + offs_d * stride_bd
        out_ptrs = Out_ptr + pid_b * stride_ob + l_idx * stride_ol + offs_d * stride_od

        x_val = tl.load(x_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        a_val = tl.load(a_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        b_val = tl.load(b_ptrs, mask=mask_d, other=0.0).to(tl.float32)

        # Fused decay update: s_t = exp(a_t) * s_{t-1} + b_t * x_t
        decay = tl.exp(a_val)
        running_state = decay * running_state + b_val * x_val
        
        tl.store(out_ptrs, running_state, mask=mask_d)

def triton_fused_selective_scan(x: torch.Tensor, log_a: torch.Tensor, b: torch.Tensor):
    """
    High-Performance Fused GPU Triton Chunked Scan for NVIDIA H100.
    Fuses decay exponentiation, state multiplication, and memory storage into a single pass.
    """
    B, L, D = x.shape
    out = torch.empty_like(x)
    final_s = torch.zeros(B, D, device=x.device, dtype=x.dtype)

    BLOCK_D = 64
    grid = (B, triton.cdiv(D, BLOCK_D))

    _fused_chunk_scan_kernel[grid](
        x, log_a, b, out, final_s,
        x.stride(0), x.stride(1), x.stride(2),
        log_a.stride(0), log_a.stride(1), log_a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        B, L, D,
        BLOCK_D=BLOCK_D,
        CHUNK_SIZE=32,
        num_warps=4,
        num_stages=2
    )
    return out
