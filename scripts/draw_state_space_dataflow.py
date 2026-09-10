#!/usr/bin/env python3
"""
Publication-Grade Architecture Diagram Generator (Mathematical Edition)
Hokie-LM: State Space Dynamics & Cognitive Active Forgetting (CAFE) Dataflow
Outputs high-res publication figures to figures/, paper/figures/, and presentation/figures/
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def create_diagram():
    # 16:11 Aspect Ratio at 300 DPI for crystal clear publication quality
    fig, ax = plt.subplots(figsize=(16, 11.2), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11.2)
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
    ax.text(8.0, 10.75, "Hokie-LM: State Space Dynamics & Cognitive Active Forgetting (CAFE)", 
            ha='center', va='center', fontsize=17, fontweight='bold', color=c_dark)
    ax.text(8.0, 10.38, r"Complete Mathematical Dataflow: Novelty Gating ($\gamma_t$), Multi-Scale Decay ($\Omega$), and Orthogonal Nullification ($P_\perp$)", 
            ha='center', va='center', fontsize=10.5, color=c_muted)

    # -------------------------------------------------------------------------
    # 1. TOP SECTION: INPUT TOKEN & COGNITIVE SURPRISE DETECTION
    # -------------------------------------------------------------------------
    box_top = patches.FancyBboxPatch((0.8, 7.85), 14.4, 2.25, boxstyle="round,pad=0.12",
                                     facecolor='#FAFAFA', edgecolor='#E2E8F0', linewidth=1.2)
    ax.add_patch(box_top)
    ax.text(1.2, 9.85, "Phase 1: Input Perception & Cognitive Gating Engine", fontsize=11, fontweight='bold', color=c_dark)

    # Input Token Box
    inp_box = patches.FancyBboxPatch((1.2, 8.1), 3.0, 1.45, boxstyle="round,pad=0.08",
                                     facecolor=c_input_bg, edgecolor=c_input_bdr, linewidth=1.4)
    ax.add_patch(inp_box)
    ax.text(2.7, 9.25, r"Input Token: $x_t$", ha='center', va='center', fontsize=11, fontweight='bold', color=c_dark)
    ax.text(2.7, 8.85, r"$x_t \in \mathbb{R}^{B \times d_{\mathrm{model}}}$", ha='center', va='center', fontsize=9.2, color=c_muted)
    ax.text(2.7, 8.48, r"$u = W_{\mathrm{in}} x_t \in \mathbb{R}^{d_{\mathrm{inner}}}$", ha='center', va='center', fontsize=9.2, fontweight='bold', color=c_body)
    ax.text(2.7, 8.22, r"$\Delta t = \mathrm{softplus}(W_{\Delta} u)$", ha='center', va='center', fontsize=8.5, color='#0369A1')

    # Arrow from Input to Surprise Gate & Feature
    ax.annotate("", xy=(4.7, 9.15), xytext=(4.2, 9.15), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.5))
    ax.annotate("", xy=(4.7, 8.35), xytext=(4.2, 8.35), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.5))

    # Surprise Gate Box
    surp_box = patches.FancyBboxPatch((4.7, 8.1), 4.3, 1.45, boxstyle="round,pad=0.08",
                                      facecolor=c_surprise_bg, edgecolor=c_surprise_bdr, linewidth=1.4)
    ax.add_patch(surp_box)
    ax.text(6.85, 9.25, r"Surprise Gate: $\gamma_t = \mathcal{D}_{\mathrm{KL}}(q_\phi \parallel p_\theta)$", ha='center', va='center',
            fontsize=10.2, fontweight='bold', color='#92400E')
    ax.text(6.85, 8.80, r"$\gamma_t = \sum_k q(z_k) \cdot [ \log q(z_k) - \log p(z_k) ]$", ha='center', va='center',
            fontsize=8.8, fontweight='bold', color='#78350F')
    ax.text(6.85, 8.35, "Measures informational novelty / unexpectedness", ha='center', va='center',
            fontsize=7.8, color='#78350F')

    # Salience & Eviction Gates
    # Storage Salience Gate
    sal_box = patches.FancyBboxPatch((9.6, 8.95), 5.2, 0.85, boxstyle="round,pad=0.06",
                                     facecolor=c_gate_bg, edgecolor=c_gate_bdr, linewidth=1.2)
    ax.add_patch(sal_box)
    ax.text(12.2, 9.55, r"Storage Salience Gate: $G_{\mathrm{write}} = \sigma(W_s \gamma_t)$", ha='center', va='center',
            fontsize=9.5, fontweight='bold', color=c_dark)
    ax.text(12.2, 9.20, r"$u_{\mathrm{eff}} = u \odot (1.0 + G_{\mathrm{write}}) \quad \rightarrow \text{Boost Write Up to } 2.0\times$", ha='center', va='center',
            fontsize=8.2, fontweight='bold', color='#0369A1')

    # Eviction Gate
    evict_box = patches.FancyBboxPatch((9.6, 7.95), 5.2, 0.85, boxstyle="round,pad=0.06",
                                       facecolor=c_gate_bg, edgecolor=c_gate_bdr, linewidth=1.2)
    ax.add_patch(evict_box)
    ax.text(12.2, 8.55, r"Active Eviction Gate: $E_t \in [0, 1]^{d_{\mathrm{inner}}}$", ha='center', va='center',
            fontsize=9.5, fontweight='bold', color=c_dark)
    ax.text(12.2, 8.20, r"$E_t = \sigma(W_e u - \beta \gamma_t + \mathrm{Bias}) \quad \rightarrow \text{Protects via } (-\beta \gamma_t)$", ha='center', va='center',
            fontsize=8.0, fontweight='bold', color='#B91C1C')

    # Connect Surprise to Gates
    ax.annotate("", xy=(9.6, 9.35), xytext=(9.0, 9.0), arrowprops=dict(arrowstyle="->", color=c_surprise_bdr, lw=1.3))
    ax.annotate("", xy=(9.6, 8.40), xytext=(9.0, 8.7), arrowprops=dict(arrowstyle="->", color=c_surprise_bdr, lw=1.3))

    # -------------------------------------------------------------------------
    # 2. MIDDLE SECTION: MULTI-SCALE CHANNEL LIFETIMES (Ω PARTITIONING)
    # -------------------------------------------------------------------------
    box_mid = patches.FancyBboxPatch((0.8, 4.85), 14.4, 2.75, boxstyle="round,pad=0.12",
                                     facecolor='#FAFAFA', edgecolor='#E2E8F0', linewidth=1.2)
    ax.add_patch(box_mid)
    ax.text(1.2, 7.35, r"Phase 2: Multi-Scale Channel Lifetimes & Modulated Decay ($\Omega$ Multiplier)", 
            fontsize=11, fontweight='bold', color=c_dark)

    # Eviction Gate downward arrow
    ax.annotate(r"Eviction Signal $E_t$", xy=(8.0, 6.95), xytext=(12.2, 7.95),
                arrowprops=dict(arrowstyle="->", color='#B91C1C', lw=1.4, linestyle='--'),
                fontsize=8.5, fontweight='bold', color='#B91C1C', ha='center')

    # 3 Channels Columns
    # Channel 1: Persistent
    ch1_box = patches.FancyBboxPatch((1.2, 5.15), 4.2, 1.70, boxstyle="round,pad=0.08",
                                     facecolor=c_pers_bg, edgecolor=c_pers_bdr, linewidth=1.4)
    ax.add_patch(ch1_box)
    ax.text(3.3, 6.55, r"1. Persistent Channels ($60\%$ Slots)", ha='center', va='center',
            fontsize=9.8, fontweight='bold', color=c_pers_bdr)
    ax.text(3.3, 6.20, r"$\Omega_{\mathrm{pers}} = 0.05 \quad \rightarrow \quad \Delta t_{\mathrm{eff}} \approx \Delta t$", ha='center', va='center',
            fontsize=8.8, fontweight='bold', color=c_dark)
    ax.text(3.3, 5.75, "• Invariant rules, System Prompt, Persona\n• Near-zero decay across 100k+ tokens\n• Immune to catastrophic amnesia",
            ha='center', va='center', fontsize=7.6, color=c_body)

    # Channel 2: Working Memory
    ch2_box = patches.FancyBboxPatch((5.9, 5.15), 4.2, 1.70, boxstyle="round,pad=0.08",
                                     facecolor=c_work_bg, edgecolor=c_work_bdr, linewidth=1.4)
    ax.add_patch(ch2_box)
    ax.text(8.0, 6.55, r"2. Working Memory Channels ($30\%$ Slots)", ha='center', va='center',
            fontsize=9.8, fontweight='bold', color=c_work_bdr)
    ax.text(8.0, 6.20, r"$\Omega_{\mathrm{work}} = 1.00 \quad \rightarrow \quad \Delta t_{\mathrm{eff}} = \Delta t (1 + E_t)$", ha='center', va='center',
            fontsize=8.8, fontweight='bold', color=c_dark)
    ax.text(8.0, 5.75, "• Dynamic contextual dialogue flow\n• Topic-boundary aware smooth flush\n• Balances retention with fresh updates",
            ha='center', va='center', fontsize=7.6, color=c_body)

    # Channel 3: Ephemeral Scratchpad
    ch3_box = patches.FancyBboxPatch((10.6, 5.15), 4.2, 1.70, boxstyle="round,pad=0.08",
                                     facecolor=c_scratch_bg, edgecolor=c_scratch_bdr, linewidth=1.4)
    ax.add_patch(ch3_box)
    ax.text(12.7, 6.55, r"3. Ephemeral Scratchpad ($10\%$ Slots)", ha='center', va='center',
            fontsize=9.8, fontweight='bold', color=c_scratch_bdr)
    ax.text(12.7, 6.20, r"$\Omega_{\mathrm{scratch}} = 25.0 \quad \rightarrow \quad \Delta t_{\mathrm{eff}} = \Delta t (1 + 25 E_t)$", ha='center', va='center',
            fontsize=8.8, fontweight='bold', color=c_dark)
    ax.text(12.7, 5.75, "• Scratchpad arithmetic, typos, nonce tokens\n• Ultra-fast eviction to 0.0000 after step\n• Eliminates distractor noise (Context Rot)",
            ha='center', va='center', fontsize=7.6, color=c_body)

    # Modulated Effective Step Formula Banner
    ax.text(8.0, 4.95, r"Master Channel Decay: $\Delta t_{\mathrm{eff}}^{(i)} = \Delta t^{(i)} \cdot ( 1 + \Omega_i \cdot E_t^{(i)} ) \quad \rightarrow \quad \bar{A} = \exp( -\Delta t_{\mathrm{eff}} A )$",
            ha='center', va='center', fontsize=9.2, fontweight='bold', color='#1E293B',
            bbox=dict(boxstyle="round,pad=0.25", facecolor='#F1F5F9', edgecolor='#94A3B8', lw=1.0))

    # -------------------------------------------------------------------------
    # 3. BOTTOM SECTION: STATE UPDATE, ORTHOGONAL NULLIFICATION & EMISSION
    # -------------------------------------------------------------------------
    box_bot = patches.FancyBboxPatch((0.8, 0.45), 14.4, 4.15, boxstyle="round,pad=0.12",
                                     facecolor='#FAFAFA', edgecolor='#E2E8F0', linewidth=1.2)
    ax.add_patch(box_bot)
    ax.text(1.2, 4.35, r"Phase 3: State Space Recurrence & Orthogonal Nullification ($O(1) = 17.0\mathrm{ KB}$ Footprint)", 
            fontsize=11, fontweight='bold', color=c_dark)

    # State Core Block
    core_box = patches.FancyBboxPatch((1.2, 2.25), 8.8, 1.85, boxstyle="round,pad=0.08",
                                      facecolor=c_core_bg, edgecolor=c_core_bdr, linewidth=1.4)
    ax.add_patch(core_box)
    ax.text(5.6, 3.82, r"Dual State Space Recurrence Core ($O(1)$ Constant Memory: Exact $17.0\mathrm{ KB}$)", 
            ha='center', va='center', fontsize=10.2, fontweight='bold', color=c_core_bdr)

    # Mathematical Equation inside Core
    ax.text(5.6, 3.38, r"$h_t^{\mathrm{raw}} = \exp( -\Delta t_{\mathrm{eff}} \odot A ) \odot h_{t-1} + ( \Delta t \cdot B_t ) \odot u_{\mathrm{eff}}$", 
            ha='center', va='center', fontsize=10.2, fontweight='bold', color=c_dark)
    ax.text(5.6, 2.92, r"Continuous SSM: $h_t \in \mathbb{R}^{2 \times 128 \times 16}$ ($16.0\mathrm{ KB}$)  $+$  Discrete Belief: $z_t \sim q_\phi(z \mid h_t, x_t)$ ($1.0\mathrm{ KB}$)", 
            ha='center', va='center', fontsize=8.2, color=c_muted)
    ax.text(5.6, 2.55, r"$h_t$ acts as a continuous Associative Hash Map: binds keys and values via Outer Products", 
            ha='center', va='center', fontsize=7.8, fontweight='bold', color='#16A34A')

    # Connect Phase 2 to Core
    ax.annotate("", xy=(5.6, 4.15), xytext=(5.6, 4.75), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.5))

    # Orthogonal Nullification Block
    proj_box = patches.FancyBboxPatch((10.4, 2.25), 4.4, 1.85, boxstyle="round,pad=0.08",
                                      facecolor=c_proj_bg, edgecolor=c_proj_bdr, linewidth=1.4)
    ax.add_patch(proj_box)
    ax.text(12.6, 3.82, r"Orthogonal Nullification ($P_\perp$)", ha='center', va='center',
            fontsize=10.0, fontweight='bold', color=c_proj_bdr)
    ax.text(12.6, 3.38, r"$h_t \leftarrow P_\perp \cdot h_t^{\mathrm{raw}}$", ha='center', va='center',
            fontsize=10.2, fontweight='bold', color=c_dark)
    ax.text(12.6, 2.95, r"$P_\perp = I - V_{\mathrm{del}} V_{\mathrm{del}}^\top$", ha='center', va='center',
            fontsize=9.0, fontweight='bold', color=c_body)
    ax.text(12.6, 2.55, "Surgically zeroes target subspace\n(0.0000% leakage verification on unlearning)", 
            ha='center', va='center', fontsize=7.2, color='#B91C1C')

    # Arrow from Core to Orthogonal Nullification
    ax.annotate("", xy=(10.4, 3.17), xytext=(10.0, 3.17), arrowprops=dict(arrowstyle="->", color=c_proj_bdr, lw=1.5))

    # Output Generation & Next Step Loops
    # Output Block
    out_box = patches.FancyBboxPatch((1.2, 0.65), 6.4, 1.35, boxstyle="round,pad=0.08",
                                     facecolor=c_out_bg, edgecolor=c_out_bdr, linewidth=1.4)
    ax.add_patch(out_box)
    ax.text(4.4, 1.70, "Output Emission & Next Token Logits", ha='center', va='center',
            fontsize=9.8, fontweight='bold', color=c_out_bdr)
    ax.text(4.4, 1.32, r"$y_t = (C_t h_t + D x_t) \odot \mathrm{SiLU}(\mathrm{gate}_t) \quad \rightarrow \quad \mathrm{Logits}_t = W_v y_t$", ha='center', va='center',
            fontsize=8.0, fontweight='bold', color=c_dark)
    ax.text(4.4, 0.92, r"Zero-Token Latent Rollout: $Q(\tau) = \sum_{\tau=1}^K \lambda^\tau V(z_\tau, h_\tau)$ (No Text Needed)", 
            ha='center', va='center', fontsize=7.2, color=c_muted)

    # Carryover Block
    carry_box = patches.FancyBboxPatch((8.4, 0.65), 6.4, 1.35, boxstyle="round,pad=0.08",
                                       facecolor='#F8FAFC', edgecolor='#64748B', linewidth=1.4)
    ax.add_patch(carry_box)
    ax.text(11.6, 1.70, r"Next Step State Recurrence ($h_t \rightarrow h_{t-1}^{(t+1)}$)", ha='center', va='center',
            fontsize=9.8, fontweight='bold', color=c_dark)
    ax.text(11.6, 1.32, r"$h_t$ carries forward directly as initial state for step $t+1$", ha='center', va='center',
            fontsize=8.5, fontweight='bold', color=c_body)
    ax.text(11.6, 0.92, r"Flat $17.0\mathrm{ KB}$ memory is maintained for $L \geq 100\mathrm{k}$ (Zero KV Cache Growth)", 
            ha='center', va='center', fontsize=7.5, fontweight='bold', color='#0284C7')

    # Connect Core / Projection to Outputs
    ax.annotate("", xy=(4.4, 2.0), xytext=(4.4, 2.25), arrowprops=dict(arrowstyle="->", color=c_out_bdr, lw=1.5))
    ax.annotate("", xy=(11.6, 2.0), xytext=(12.6, 2.25), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.5))

    # Save to all target paths
    os.makedirs("figures", exist_ok=True)
    os.makedirs("paper/figures", exist_ok=True)
    os.makedirs("presentation/figures", exist_ok=True)

    out_path1 = "figures/fig19_state_space_cafe_dataflow.png"
    out_path2 = "paper/figures/fig19_state_space_cafe_dataflow.png"
    out_path3 = "presentation/figures/fig19_state_space_cafe_dataflow.png"

    plt.savefig(out_path1, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.savefig(out_path2, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.savefig(out_path3, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()

    print(f"✅ Generated with complete LaTeX math: {out_path1}")
    print(f"✅ Generated with complete LaTeX math: {out_path2}")
    print(f"✅ Generated with complete LaTeX math: {out_path3}")

if __name__ == "__main__":
    create_diagram()
