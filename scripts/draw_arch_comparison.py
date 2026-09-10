#!/usr/bin/env python3
"""
Publication-Grade Architecture Diagram Generator (Clean White Edition)
Standard Transformer Layer vs NeuroWorld-LM Layer with CAFE
Designed with minimal, elegant, academic styling on pure white background.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil
import os

def build_clean_white_diagram():
    # 16:7.5 Widescreen Aspect Ratio at 300 DPI
    fig, ax = plt.subplots(figsize=(16, 7.5), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.5)
    ax.axis('off')

    # Typography & Palette (Minimalist Academic)
    c_text_dark = '#0F172A'       # Slate 900
    c_text_body = '#1E293B'       # Slate 800
    c_text_muted = '#64748B'      # Slate 500
    c_line = '#94A3B8'            # Slate 400
    c_arrow = '#475569'           # Slate 600

    # Transformer Theme (Subtle Rose/Crimson Accents)
    c_tf_border = '#E11D48'
    c_tf_bg = '#FFF1F2'
    c_tf_card = '#FFFFFF'
    c_tf_kv = '#FEE2E2'
    c_tf_kv_border = '#DC2626'

    # NeuroWorld Theme (Subtle Cyan/Emerald/Purple Accents)
    c_nw_border = '#0284C7'
    c_nw_bg = '#F0F9FF'
    c_nw_card = '#FFFFFF'
    c_ssm_bg = '#E0F2FE'
    c_ssm_border = '#0284C7'
    c_cafe_bg = '#ECFDF5'
    c_cafe_border = '#059669'
    c_rssm_bg = '#F5F3FF'
    c_rssm_border = '#7C3AED'
    c_rollout_bg = '#F0FDF4'
    c_rollout_border = '#16A34A'

    # -------------------------------------------------------------
    # TOP HEADER
    # -------------------------------------------------------------
    ax.text(8.0, 7.15, "Layer Architecture: Standard Transformer vs. NeuroWorld-LM", 
            ha='center', va='center', fontsize=17, fontweight='bold', color=c_text_dark)
    ax.text(8.0, 6.82, "Detailed computational dataflow, memory buffers, and state recurrence inside a single layer l", 
            ha='center', va='center', fontsize=10.5, color=c_text_muted)

    # -------------------------------------------------------------
    # LEFT PANEL: Transformer Layer l
    # -------------------------------------------------------------
    # Outer Layer Boundary Box
    tf_outer = patches.FancyBboxPatch((0.8, 0.45), 6.9, 6.1, boxstyle="round,pad=0.15",
                                      facecolor='#FAFAFA', edgecolor='#CBD5E1', linewidth=1.5)
    ax.add_patch(tf_outer)

    # Stack Bracket: Layer (l)
    ax.text(1.15, 3.5, "Stacked  x L Layers", ha='center', va='center', rotation=90,
            fontsize=9.5, fontweight='bold', color=c_tf_border)
    ax.plot([1.35, 1.35], [1.0, 6.0], color='#CBD5E1', lw=1.2, linestyle='--')

    # Panel Title
    ax.text(4.45, 6.25, "Standard Transformer Layer (l)", ha='center', va='center',
            fontsize=12.5, fontweight='bold', color=c_tf_border)
    ax.text(4.45, 6.02, "Autoregressive Multi-Head Self-Attention with KV Cache", ha='center', va='center',
            fontsize=8.5, color=c_text_muted)

    # Input Tensor
    inp_tf = patches.FancyBboxPatch((2.65, 5.48), 3.6, 0.35, boxstyle="round,pad=0.06",
                                    facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=1.0)
    ax.add_patch(inp_tf)
    ax.text(4.45, 5.655, "Input:  x_t^(l-1)  in  R^(B x d_model)", ha='center', va='center',
            fontsize=8.8, fontweight='bold', color=c_text_body)

    # Arrow Down to RMSNorm 1
    ax.annotate("", xy=(4.45, 5.2), xytext=(4.45, 5.48), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.2))

    # RMSNorm 1
    norm1_tf = patches.FancyBboxPatch((3.45, 4.95), 2.0, 0.25, boxstyle="round,pad=0.04",
                                     facecolor='#F1F5F9', edgecolor='#94A3B8', linewidth=0.8)
    ax.add_patch(norm1_tf)
    ax.text(4.45, 5.075, "RMSNorm", ha='center', va='center', fontsize=8.0, color=c_text_body)

    # Arrow Down to Attention Core
    ax.annotate("", xy=(4.45, 4.65), xytext=(4.45, 4.95), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.2))

    # Multi-Head Attention Core
    mha_box = patches.FancyBboxPatch((1.65, 3.0), 5.6, 1.65, boxstyle="round,pad=0.1",
                                     facecolor='#FFF1F2', edgecolor='#FDA4AF', linewidth=1.2)
    ax.add_patch(mha_box)
    ax.text(4.45, 4.45, "Multi-Head Self-Attention (MHA / GQA)", ha='center', va='center',
            fontsize=9.8, fontweight='bold', color=c_tf_border)

    # Q, K, V Projections
    ax.text(2.45, 4.12, "Q = x W_Q", ha='center', va='center', fontsize=7.8, color=c_text_dark,
            bbox=dict(boxstyle="round,pad=0.15", facecolor='#FFFFFF', edgecolor='#FDA4AF'))
    ax.text(3.78, 4.12, "K = x W_K", ha='center', va='center', fontsize=7.8, color=c_text_dark,
            bbox=dict(boxstyle="round,pad=0.15", facecolor='#FFFFFF', edgecolor='#FDA4AF'))
    ax.text(5.12, 4.12, "V = x W_V", ha='center', va='center', fontsize=7.8, color=c_text_dark,
            bbox=dict(boxstyle="round,pad=0.15", facecolor='#FFFFFF', edgecolor='#FDA4AF'))

    # KV-Cache Pool Box (Expanding Buffer)
    kv_box = patches.FancyBboxPatch((1.85, 3.15), 5.2, 0.75, boxstyle="round,pad=0.06",
                                    facecolor=c_tf_kv, edgecolor=c_tf_kv_border, linewidth=1.2, linestyle='--')
    ax.add_patch(kv_box)
    ax.text(4.45, 3.7, "[!] Key-Value Cache Buffer (Unbounded O(T) VRAM)", ha='center', va='center',
            fontsize=8.2, fontweight='bold', color=c_tf_kv_border)
    ax.text(4.45, 3.48, "History Buffer: K_{<=t} = [k_1, ..., k_t],  V_{<=t} = [v_1, ..., v_t]", ha='center', va='center',
            fontsize=7.4, color=c_text_body)
    ax.text(4.45, 3.28, "Softmax Weights: w_ij = exp(q_i k_j^T / sqrt(d)) > 0  ==>  NO FORGETTING", ha='center', va='center',
            fontsize=7.2, fontweight='bold', color='#991B1B')

    # Arrow Down to Add 1
    ax.annotate("", xy=(4.45, 2.7), xytext=(4.45, 3.0), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.2))

    # Add 1 Circle
    add1 = patches.Circle((4.45, 2.58), 0.11, facecolor='#FFFFFF', edgecolor=c_tf_border, lw=1.2)
    ax.add_patch(add1)
    ax.text(4.45, 2.58, "+", ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_tf_border)

    # Residual Connection 1
    ax.plot([2.1, 2.1, 4.34], [5.655, 2.58, 2.58], color='#E11D48', lw=1.0, linestyle=':')

    # Arrow Down to RMSNorm 2
    ax.annotate("", xy=(4.45, 2.3), xytext=(4.45, 2.47), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.2))

    # RMSNorm 2
    norm2_tf = patches.FancyBboxPatch((3.45, 2.08), 2.0, 0.22, boxstyle="round,pad=0.04",
                                     facecolor='#F1F5F9', edgecolor='#94A3B8', linewidth=0.8)
    ax.add_patch(norm2_tf)
    ax.text(4.45, 2.19, "RMSNorm", ha='center', va='center', fontsize=8.0, color=c_text_body)

    # Arrow Down to SwiGLU FFN
    ax.annotate("", xy=(4.45, 1.8), xytext=(4.45, 2.08), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.2))

    # SwiGLU FFN Box
    ffn_box = patches.FancyBboxPatch((2.15, 1.3), 4.6, 0.5, boxstyle="round,pad=0.08",
                                     facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=1.0)
    ax.add_patch(ffn_box)
    ax.text(4.45, 1.6, "Feed-Forward Network (SwiGLU / MLP)", ha='center', va='center',
            fontsize=8.8, fontweight='bold', color=c_text_dark)
    ax.text(4.45, 1.42, "y = (SiLU(x W_gate) * (x W_up)) W_down", ha='center', va='center',
            fontsize=7.4, color=c_text_muted)

    # Arrow Down to Add 2
    ax.annotate("", xy=(4.45, 1.05), xytext=(4.45, 1.3), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.2))

    # Add 2 Circle
    add2 = patches.Circle((4.45, 0.94), 0.11, facecolor='#FFFFFF', edgecolor=c_tf_border, lw=1.2)
    ax.add_patch(add2)
    ax.text(4.45, 0.94, "+", ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_tf_border)

    # Residual Connection 2
    ax.plot([2.3, 2.3, 4.34], [2.58, 0.94, 0.94], color='#E11D48', lw=1.0, linestyle=':')

    # Arrow to Output
    ax.annotate("", xy=(4.45, 0.72), xytext=(4.45, 0.83), arrowprops=dict(arrowstyle="->", color=c_tf_border, lw=1.2))

    # Output Box
    out_tf = patches.FancyBboxPatch((2.65, 0.52), 3.6, 0.28, boxstyle="round,pad=0.06",
                                    facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=1.0)
    ax.add_patch(out_tf)
    ax.text(4.45, 0.66, "Layer Output:  x_t^(l)  ->  Layer (l+1)", ha='center', va='center',
            fontsize=8.5, fontweight='bold', color=c_text_body)

    # Bottom Tag
    ax.text(4.45, 0.22, "Characteristics: O(T) Memory Growth | Attention Softmax Prevents Forgetting",
            ha='center', va='center', fontsize=8.0, fontweight='bold', color=c_tf_kv_border)

    # -------------------------------------------------------------
    # RIGHT PANEL: NeuroWorld-LM Layer l with CAFE
    # -------------------------------------------------------------
    # Outer Layer Boundary Box
    nw_outer = patches.FancyBboxPatch((8.3, 0.45), 6.9, 6.1, boxstyle="round,pad=0.15",
                                      facecolor='#FAFAFA', edgecolor='#BAE6FD', linewidth=1.5)
    ax.add_patch(nw_outer)

    # Stack Bracket: Layer (l)
    ax.text(8.65, 3.5, "Stacked  x L Layers", ha='center', va='center', rotation=90,
            fontsize=9.5, fontweight='bold', color=c_nw_border)
    ax.plot([8.85, 8.85], [1.0, 6.0], color='#BAE6FD', lw=1.2, linestyle='--')

    # Panel Title
    ax.text(11.95, 6.25, "NeuroWorld-LM World Model Layer (l)", ha='center', va='center',
            fontsize=12.5, fontweight='bold', color=c_nw_border)
    ax.text(11.95, 6.02, "Constant-Memory Dual-State with Cognitive Active Forgetting Engine", ha='center', va='center',
            fontsize=8.5, color=c_text_muted)

    # Recurrent Dual-State Inputs (Top Left/Right)
    h_in = patches.FancyBboxPatch((9.15, 5.48), 2.2, 0.35, boxstyle="round,pad=0.06",
                                  facecolor=c_ssm_bg, edgecolor=c_ssm_border, linewidth=1.0)
    ax.add_patch(h_in)
    ax.text(10.25, 5.655, "h_{t-1}^(l): 17KB SSM State", ha='center', va='center',
            fontsize=7.8, fontweight='bold', color=c_ssm_border)

    z_in = patches.FancyBboxPatch((12.6, 5.48), 2.2, 0.35, boxstyle="round,pad=0.06",
                                  facecolor=c_rssm_bg, edgecolor=c_rssm_border, linewidth=1.0)
    ax.add_patch(z_in)
    ax.text(13.7, 5.655, "z_{t-1}^(l): Latent Belief", ha='center', va='center',
            fontsize=7.8, fontweight='bold', color=c_rssm_border)

    # Layer Input x_t^(l-1) (Top Center)
    inp_nw = patches.FancyBboxPatch((10.7, 5.08), 2.5, 0.28, boxstyle="round,pad=0.04",
                                    facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=0.8)
    ax.add_patch(inp_nw)
    ax.text(11.95, 5.22, "Input:  x_t^(l-1)", ha='center', va='center',
            fontsize=8.0, fontweight='bold', color=c_text_body)

    # Arrows to CAFE Core
    ax.annotate("", xy=(10.25, 4.65), xytext=(10.25, 5.48), arrowprops=dict(arrowstyle="->", color=c_ssm_border, lw=1.2))
    ax.annotate("", xy=(11.95, 4.65), xytext=(11.95, 5.08), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.2))
    ax.annotate("", xy=(13.7, 4.65), xytext=(13.7, 5.48), arrowprops=dict(arrowstyle="->", color=c_rssm_border, lw=1.2))

    # MODULE 1: Cognitive Active Forgetting Engine (CAFE)
    cafe_box = patches.FancyBboxPatch((9.15, 3.5), 5.7, 1.15, boxstyle="round,pad=0.08",
                                      facecolor=c_cafe_bg, edgecolor=c_cafe_border, linewidth=1.2)
    ax.add_patch(cafe_box)
    ax.text(12.0, 4.45, "[1] Cognitive Active Forgetting Engine (CAFE)", ha='center', va='center',
            fontsize=9.5, fontweight='bold', color=c_cafe_border)

    ax.text(10.5, 4.12, "Surprise: γ_t = D_KL(q || p)", ha='center', va='center', fontsize=7.6, color=c_text_dark,
            bbox=dict(boxstyle="round,pad=0.15", facecolor='#FFFFFF', edgecolor=c_cafe_border))
    ax.text(13.5, 4.12, "Eviction: E_t ⊙ h_{t-1}", ha='center', va='center', fontsize=7.6, color='#991B1B',
            bbox=dict(boxstyle="round,pad=0.15", facecolor='#FEE2E2', edgecolor='#EF4444'))

    null_banner = patches.FancyBboxPatch((9.35, 3.62), 5.3, 0.32, boxstyle="round,pad=0.04",
                                         facecolor='#CCFBF1', edgecolor='#0D9488', linewidth=0.8)
    ax.add_patch(null_banner)
    ax.text(12.0, 3.78, "Subspace Nullifier:  P_perp = I - sum(q_k q_k^T)  ==>  0.00% Zero-Leakage",
            ha='center', va='center', fontsize=7.4, fontweight='bold', color='#0F766E')

    # Arrow to State Core
    ax.annotate("", xy=(12.0, 3.18), xytext=(12.0, 3.5), arrowprops=dict(arrowstyle="->", color=c_cafe_border, lw=1.2))

    # MODULE 2: Dual-Loop State Core (SSM + Categorical RSSM)
    dual_core = patches.FancyBboxPatch((9.15, 1.8), 5.7, 1.38, boxstyle="round,pad=0.1",
                                       facecolor=c_ssm_bg, edgecolor=c_ssm_border, linewidth=1.2)
    ax.add_patch(dual_core)
    ax.text(12.0, 3.0, "[2] Dual-Loop State-Space Core (Selective SSM + Categorical RSSM)", ha='center', va='center',
            fontsize=9.0, fontweight='bold', color=c_ssm_border)

    ssm_sub = patches.FancyBboxPatch((9.35, 2.38), 5.3, 0.45, boxstyle="round,pad=0.06",
                                     facecolor='#FFFFFF', edgecolor='#7DD3FC', linewidth=0.8)
    ax.add_patch(ssm_sub)
    ax.text(12.0, 2.65, "Continuous SSM:  h_t^(l) = P_perp [ A_t h_{t-1} - E_t ⊙ h_{t-1} ] + B~_t x_t",
            ha='center', va='center', fontsize=7.4, fontweight='bold', color='#0369A1')
    ax.text(12.0, 2.48, "State Buffer: Strictly Constant 17.0 KB  (1,927x Memory Reduction over KV Cache)",
            ha='center', va='center', fontsize=7.0, color='#D97706')

    rssm_sub = patches.FancyBboxPatch((9.35, 1.9), 5.3, 0.4, boxstyle="round,pad=0.06",
                                      facecolor='#FAF5FF', edgecolor='#C084FC', linewidth=0.8)
    ax.add_patch(rssm_sub)
    ax.text(12.0, 2.15, "Categorical Latents:  z_t^(l) ~ q_phi(z_t | h_t, x_t)  vs  p_theta(z_t | h_t)",
            ha='center', va='center', fontsize=7.4, fontweight='bold', color='#6B21A8')
    ax.text(12.0, 1.98, "Stochastic Multi-Hypothesis World Beliefs (Zero Collapse)",
            ha='center', va='center', fontsize=6.8, color=c_text_muted)

    # Arrow to Rollout Planner
    ax.annotate("", xy=(12.0, 1.5), xytext=(12.0, 1.8), arrowprops=dict(arrowstyle="->", color=c_ssm_border, lw=1.2))

    # MODULE 3: Zero-Token Latent Rollout Planner
    rollout_box = patches.FancyBboxPatch((9.35, 1.1), 5.3, 0.4, boxstyle="round,pad=0.06",
                                         facecolor=c_rollout_bg, edgecolor=c_rollout_border, linewidth=1.0)
    ax.add_patch(rollout_box)
    ax.text(12.0, 1.34, "[3] Zero-Token Latent Rollout Planner (Mental Simulation)", ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='#15803D')
    ax.text(12.0, 1.18, "Value-Head Guided Rollout directly in (h, z)  |  12x FLOPs Saved (Zero Token Overhead)",
            ha='center', va='center', fontsize=7.0, color=c_text_body)

    # Arrow to Output
    ax.annotate("", xy=(12.0, 0.82), xytext=(12.0, 1.1), arrowprops=dict(arrowstyle="->", color=c_ssm_border, lw=1.2))

    # Output Box
    out_nw = patches.FancyBboxPatch((10.2, 0.52), 3.6, 0.28, boxstyle="round,pad=0.06",
                                    facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=1.0)
    ax.add_patch(out_nw)
    ax.text(12.0, 0.66, "Layer Output:  x_t^(l)  ->  Layer (l+1)", ha='center', va='center',
            fontsize=8.5, fontweight='bold', color=c_text_body)

    # Bottom Tag
    ax.text(11.95, 0.22, "Characteristics: O(1) Constant 17KB Memory | Subspace Projection P_perp | Latent Rollout",
            ha='center', va='center', fontsize=8.0, fontweight='bold', color='#0369A1')

    # Save to all figure directories
    out_file = "/home/eun/neuroworld_lm/figures/fig16_arch_layer_comparison.png"
    plt.savefig(out_file, bbox_inches='tight', facecolor='#FFFFFF', edgecolor='none', dpi=300)
    plt.close()

    # Fast file copies
    shutil.copyfile(out_file, "/home/eun/neuroworld_lm/paper/figures/fig16_arch_layer_comparison.png")
    shutil.copyfile(out_file, "/home/eun/neuroworld_lm/presentation/figures/fig16_arch_layer_comparison.png")
    print("[SUCCESS] Pristine white architecture diagram generated and deployed!")

if __name__ == "__main__":
    build_clean_white_diagram()
