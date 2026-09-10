#!/usr/bin/env python3
"""
Publication-Grade Architecture Diagram Generator
Hokie-LM: State Space Dynamics & Cognitive Active Forgetting (CAFE) Dataflow
Outputs high-res publication figures to figures/ and paper/figures/
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil
import os

def create_diagram():
    # 16:9 Widescreen Aspect Ratio at 300 DPI
    fig, ax = plt.subplots(figsize=(16, 10.5), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10.5)
    ax.axis('off')

    # Minimalist Academic Color Palette
    c_dark = '#0F172A'          # Slate 900
    c_body = '#1E293B'          # Slate 800
    c_muted = '#64748B'         # Slate 500
    c_border = '#CBD5E1'        # Slate 300
    c_arrow = '#334155'         # Slate 700

    # Thematic Color Accents
    c_input_bg = '#F8FAFC'
    c_input_bdr = '#64748B'

    c_surprise_bg = '#FEF3C7'   # Amber 100
    c_surprise_bdr = '#D97706'  # Amber 600

    c_gate_bg = '#F1F5F9'       # Slate 100
    c_gate_bdr = '#475569'      # Slate 600

    c_pers_bg = '#ECFDF5'       # Emerald 50
    c_pers_bdr = '#059669'      # Emerald 600
    c_work_bg = '#EFF6FF'       # Blue 50
    c_work_bdr = '#2563EB'      # Blue 600
    c_scratch_bg = '#FFF1F2'    # Rose 50
    c_scratch_bdr = '#E11D48'   # Rose 600

    c_core_bg = '#FAF5FF'       # Purple 50
    c_core_bdr = '#7C3AED'      # Purple 600

    c_proj_bg = '#FDF4FF'       # Fuchsia 50
    c_proj_bdr = '#C026D3'      # Fuchsia 600

    c_out_bg = '#F0FDF4'        # Green 50
    c_out_bdr = '#16A34A'       # Green 600

    # -------------------------------------------------------------------------
    # HEADER SECTION
    # -------------------------------------------------------------------------
    ax.text(8.0, 10.05, "Hokie-LM: State Space Dynamics & Cognitive Active Forgetting (CAFE)", 
            ha='center', va='center', fontsize=17, fontweight='bold', color=c_dark)
    ax.text(8.0, 9.68, "Novelty-guided writing (γ_t), multi-scale channel eviction (Ω), and orthogonal nullification (P_⊥) in O(1) constant memory (17.0 KB)", 
            ha='center', va='center', fontsize=10.5, color=c_muted)

    # -------------------------------------------------------------------------
    # 1. TOP SECTION: INPUT TOKEN & COGNITIVE SURPRISE DETECTION
    # -------------------------------------------------------------------------
    # Outer container for Input & Gating
    box_top = patches.FancyBboxPatch((0.8, 7.35), 14.4, 2.05, boxstyle="round,pad=0.12",
                                     facecolor='#FAFAFA', edgecolor='#E2E8F0', linewidth=1.2)
    ax.add_patch(box_top)
    ax.text(1.2, 9.15, "Phase 1: Input Perception & Cognitive Gating", fontsize=11, fontweight='bold', color=c_dark)

    # Input Token Box
    inp_box = patches.FancyBboxPatch((1.2, 7.6), 2.8, 1.2, boxstyle="round,pad=0.08",
                                     facecolor=c_input_bg, edgecolor=c_input_bdr, linewidth=1.4)
    ax.add_patch(inp_box)
    ax.text(2.6, 8.4, "Input Token  x_t", ha='center', va='center', fontsize=11, fontweight='bold', color=c_dark)
    ax.text(2.6, 8.05, "Dimension:  R^(B × d_model)", ha='center', va='center', fontsize=8.5, color=c_muted)
    ax.text(2.6, 7.78, "Feature u = Linear(x_t)", ha='center', va='center', fontsize=8.5, color=c_body)

    # Arrow from Input to Surprise Gate & Direct Feature
    ax.annotate("", xy=(4.6, 8.45), xytext=(4.0, 8.45), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.5))
    ax.annotate("", xy=(4.6, 7.75), xytext=(4.0, 7.75), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.5))

    # Surprise Gate Box
    surp_box = patches.FancyBboxPatch((4.6, 7.95), 4.3, 1.0, boxstyle="round,pad=0.08",
                                      facecolor=c_surprise_bg, edgecolor=c_surprise_bdr, linewidth=1.4)
    ax.add_patch(surp_box)
    ax.text(6.75, 8.65, "Surprise Gate:  γ_t = D_KL( q_φ ∥ p_θ )", ha='center', va='center',
            fontsize=10.0, fontweight='bold', color='#92400E')
    ax.text(6.75, 8.22, "Measures informational novelty / unexpectedness", ha='center', va='center',
            fontsize=8.5, color='#78350F')

    # Salience & Eviction Gates
    # Storage Salience Gate
    sal_box = patches.FancyBboxPatch((9.5, 8.45), 5.3, 0.75, boxstyle="round,pad=0.06",
                                     facecolor=c_gate_bg, edgecolor=c_gate_bdr, linewidth=1.2)
    ax.add_patch(sal_box)
    ax.text(12.15, 8.95, "Storage Salience Gate:  G_write = σ( W_s · γ_t )", ha='center', va='center',
            fontsize=9.2, fontweight='bold', color=c_dark)
    ax.text(12.15, 8.65, "Amplify writing when novel:  u_eff = u ⊙ (1 + G_write)", ha='center', va='center',
            fontsize=8.2, color='#0369A1')

    # Eviction Gate
    evict_box = patches.FancyBboxPatch((9.5, 7.5), 5.3, 0.75, boxstyle="round,pad=0.06",
                                       facecolor=c_gate_bg, edgecolor=c_gate_bdr, linewidth=1.2)
    ax.add_patch(evict_box)
    ax.text(12.15, 8.0, "Active Eviction Gate:  E_t = σ( W_e·u - β·γ_t + Bias )", ha='center', va='center',
            fontsize=9.2, fontweight='bold', color=c_dark)
    ax.text(12.15, 7.7, "High surprise suppresses eviction (-β·γ_t) to protect memory", ha='center', va='center',
            fontsize=8.2, color='#B91C1C')

    # Connect Surprise to Gates
    ax.annotate("", xy=(9.5, 8.8), xytext=(8.9, 8.6), arrowprops=dict(arrowstyle="->", color=c_surprise_bdr, lw=1.3))
    ax.annotate("", xy=(9.5, 7.85), xytext=(8.9, 8.3), arrowprops=dict(arrowstyle="->", color=c_surprise_bdr, lw=1.3))

    # -------------------------------------------------------------------------
    # 2. MIDDLE SECTION: MULTI-SCALE CHANNEL LIFETIMES (Ω PARTITIONING)
    # -------------------------------------------------------------------------
    box_mid = patches.FancyBboxPatch((0.8, 4.45), 14.4, 2.65, boxstyle="round,pad=0.12",
                                     facecolor='#FAFAFA', edgecolor='#E2E8F0', linewidth=1.2)
    ax.add_patch(box_mid)
    ax.text(1.2, 6.85, "Phase 2: Multi-Scale Channel Lifetimes & Modulated Decay (Ω Multiplier)", 
            fontsize=11, fontweight='bold', color=c_dark)

    # Eviction Gate vector downward distribution
    ax.annotate("Eviction Signal E_t", xy=(8.0, 6.45), xytext=(12.15, 7.5),
                arrowprops=dict(arrowstyle="->", color='#B91C1C', lw=1.4, linestyle='--'),
                fontsize=8.5, fontweight='bold', color='#B91C1C', ha='center')

    # 3 Channels Columns
    # Channel 1: Persistent
    ch1_box = patches.FancyBboxPatch((1.2, 4.7), 4.2, 1.65, boxstyle="round,pad=0.08",
                                     facecolor=c_pers_bg, edgecolor=c_pers_bdr, linewidth=1.4)
    ax.add_patch(ch1_box)
    ax.text(3.3, 6.1, "1. Persistent Channels (60%)", ha='center', va='center',
            fontsize=10.0, fontweight='bold', color=c_pers_bdr)
    ax.text(3.3, 5.75, "Lifetime Sensitivity:  Ω = 0.05", ha='center', va='center',
            fontsize=8.8, fontweight='bold', color=c_dark)
    ax.text(3.3, 5.42, "• Invariant rules, System Prompt, User ID\n• Near-zero decay across 100k+ tokens\n• Never suffers from catastrophic amnesia",
            ha='center', va='center', fontsize=7.8, color=c_body)

    # Channel 2: Working Memory
    ch2_box = patches.FancyBboxPatch((5.9, 4.7), 4.2, 1.65, boxstyle="round,pad=0.08",
                                     facecolor=c_work_bg, edgecolor=c_work_bdr, linewidth=1.4)
    ax.add_patch(ch2_box)
    ax.text(8.0, 6.1, "2. Working Memory Channels (30%)", ha='center', va='center',
            fontsize=10.0, fontweight='bold', color=c_work_bdr)
    ax.text(8.0, 5.75, "Lifetime Sensitivity:  Ω = 1.00", ha='center', va='center',
            fontsize=8.8, fontweight='bold', color=c_dark)
    ax.text(8.0, 5.42, "• Dynamic contextual dialogue flow\n• Topic-boundary aware smooth flush\n• Balances retention with fresh updates",
            ha='center', va='center', fontsize=7.8, color=c_body)

    # Channel 3: Ephemeral Scratchpad
    ch3_box = patches.FancyBboxPatch((10.6, 4.7), 4.2, 1.65, boxstyle="round,pad=0.08",
                                     facecolor=c_scratch_bg, edgecolor=c_scratch_bdr, linewidth=1.4)
    ax.add_patch(ch3_box)
    ax.text(12.7, 6.1, "3. Ephemeral Scratchpad (10%)", ha='center', va='center',
            fontsize=10.0, fontweight='bold', color=c_scratch_bdr)
    ax.text(12.7, 5.75, "Lifetime Sensitivity:  Ω = 25.0 ~ 50.0", ha='center', va='center',
            fontsize=8.8, fontweight='bold', color=c_dark)
    ax.text(12.7, 5.42, "• Scratchpad arithmetic, typos, nonce tokens\n• Ultra-fast eviction to 0.0000 after reasoning\n• Eliminates distractor noise (Context Rot)",
            ha='center', va='center', fontsize=7.8, color=c_body)

    # Modulated Effective Step Formula Banner
    ax.text(8.0, 4.52, "Effective Time-Step Decay:   Δt_eff^(i) = Δt^(i) · ( 1 + Ω_i · E_t^(i) )   ⟹   dA = exp( -Δt_eff · A )",
            ha='center', va='center', fontsize=9.2, fontweight='bold', color='#1E293B',
            bbox=dict(boxstyle="round,pad=0.25", facecolor='#F1F5F9', edgecolor='#94A3B8', lw=1.0))

    # -------------------------------------------------------------------------
    # 3. BOTTOM SECTION: STATE UPDATE, ORTHOGONAL NULLIFICATION & EMISSION
    # -------------------------------------------------------------------------
    box_bot = patches.FancyBboxPatch((0.8, 0.45), 14.4, 3.75, boxstyle="round,pad=0.12",
                                     facecolor='#FAFAFA', edgecolor='#E2E8F0', linewidth=1.2)
    ax.add_patch(box_bot)
    ax.text(1.2, 3.95, "Phase 3: State Space Recurrence & Orthogonal Subspace Nullification (O(1) = 17.0 KB)", 
            fontsize=11, fontweight='bold', color=c_dark)

    # State Core Block
    core_box = patches.FancyBboxPatch((1.2, 2.05), 8.8, 1.65, boxstyle="round,pad=0.08",
                                      facecolor=c_core_bg, edgecolor=c_core_bdr, linewidth=1.4)
    ax.add_patch(core_box)
    ax.text(5.6, 3.45, "Dual State Space Recurrence Core  (Exact O(1) Constant Memory: 17.0 KB)", 
            ha='center', va='center', fontsize=10.2, fontweight='bold', color=c_core_bdr)

    # Mathematical Equation inside Core
    ax.text(5.6, 2.95, "h_t^(raw) = exp( -Δt_eff ⊙ A ) ⊙ h_(t-1)  +  ( Δt · B_t ) ⊙ u_eff", 
            ha='center', va='center', fontsize=9.8, fontweight='bold', color=c_dark)
    ax.text(5.6, 2.45, "Continuous SSM State: h_t ∈ R^(d_inner × d_state)  +  Discrete Categorical Belief: z_t ~ q(z|h, x)", 
            ha='center', va='center', fontsize=8.2, color=c_muted)
    ax.text(5.6, 2.20, "Memory footprint: Exact flat 17.0 KB (Zero KV Cache buffers required)", 
            ha='center', va='center', fontsize=8.0, fontweight='bold', color='#16A34A')

    # Connect Phase 2 to Core
    ax.annotate("", xy=(5.6, 3.7), xytext=(5.6, 4.3), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.5))

    # Orthogonal Nullification Block
    proj_box = patches.FancyBboxPatch((10.4, 2.05), 4.4, 1.65, boxstyle="round,pad=0.08",
                                      facecolor=c_proj_bg, edgecolor=c_proj_bdr, linewidth=1.4)
    ax.add_patch(proj_box)
    ax.text(12.6, 3.45, "Orthogonal Nullification (P_⊥)", ha='center', va='center',
            fontsize=10.0, fontweight='bold', color=c_proj_bdr)
    ax.text(12.6, 3.0, "h_t  ←  P_⊥ · h_t^(raw)", ha='center', va='center',
            fontsize=10.0, fontweight='bold', color=c_dark)
    ax.text(12.6, 2.58, "P_⊥ = ( I - V_del · V_del^T )", ha='center', va='center',
            fontsize=8.5, color=c_body)
    ax.text(12.6, 2.25, "Zero-out deleted variables & PII\n(0.0000% leakage verification)", 
            ha='center', va='center', fontsize=7.5, color='#B91C1C')

    # Arrow from Core to Orthogonal Nullification
    ax.annotate("", xy=(10.4, 2.87), xytext=(10.0, 2.87), arrowprops=dict(arrowstyle="->", color=c_proj_bdr, lw=1.5))

    # Output Generation & Next Step Loops
    # Output Block
    out_box = patches.FancyBboxPatch((1.2, 0.65), 6.4, 1.15, boxstyle="round,pad=0.08",
                                     facecolor=c_out_bg, edgecolor=c_out_bdr, linewidth=1.4)
    ax.add_patch(out_box)
    ax.text(4.4, 1.45, "Output Emission & Next Word Logits", ha='center', va='center',
            fontsize=9.8, fontweight='bold', color=c_out_bdr)
    ax.text(4.4, 1.12, "y_t = C_t · h_t + D · x_t   ⟹   Logits = Head(y_t)", ha='center', va='center',
            fontsize=8.8, fontweight='bold', color=c_dark)
    ax.text(4.4, 0.82, "Zero-Token Latent Rollout Planner: internal simulation without text generation", 
            ha='center', va='center', fontsize=7.5, color=c_muted)

    # Carryover Block
    carry_box = patches.FancyBboxPatch((8.4, 0.65), 6.4, 1.15, boxstyle="round,pad=0.08",
                                       facecolor='#F8FAFC', edgecolor='#64748B', linewidth=1.4)
    ax.add_patch(carry_box)
    ax.text(11.6, 1.45, "Next Step State Carryover (t ⟶ t+1)", ha='center', va='center',
            fontsize=9.8, fontweight='bold', color=c_dark)
    ax.text(11.6, 1.12, "h_t is forwarded as h_(t-1) for next token", ha='center', va='center',
            fontsize=8.8, fontweight='bold', color=c_body)
    ax.text(11.6, 0.82, "State size remains invariant at 17.0 KB regardless of sequence length L", 
            ha='center', va='center', fontsize=7.5, fontweight='bold', color='#0284C7')

    # Connect Core / Projection to Outputs
    ax.annotate("", xy=(4.4, 1.8), xytext=(4.4, 2.05), arrowprops=dict(arrowstyle="->", color=c_out_bdr, lw=1.5))
    ax.annotate("", xy=(11.6, 1.8), xytext=(12.6, 2.05), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.5))

    # Save to both paths
    os.makedirs("figures", exist_ok=True)
    os.makedirs("paper/figures", exist_ok=True)
    os.makedirs("presentation/figures", exist_ok=True)

    out_path1 = "figures/fig19_state_space_cafe_dataflow.png"
    out_path2 = "paper/figures/fig19_state_space_cafe_dataflow.png"
    out_path3 = "presentation/figures/fig19_state_space_cafe_dataflow.png"

    plt.tight_layout()
    plt.savefig(out_path1, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.savefig(out_path2, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.savefig(out_path3, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()

    print(f"✅ Generated: {out_path1}")
    print(f"✅ Generated: {out_path2}")
    print(f"✅ Generated: {out_path3}")

if __name__ == "__main__":
    create_diagram()
