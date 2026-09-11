#!/usr/bin/env python3
"""
Publication-Grade Architecture Diagram Generator for Hokie-LM / NeuroWorld-LM
Faithful, mathematically exact representation of our TRUE Cognitive World-Model Architecture:
1. Purely Attention-Free & Softmax-Free Model (Output Next-Token Logits directly to CrossEntropy/Top-p sampling).
2. Dual-Stream Categorical Latent World Model (Prior & Posterior Networks, Straight-Through Gumbel-Softmax, Surprise Gate KL).
3. Exact Selective Continuous State-Space Engine (SSM) with Multi-Scale State Buffers (Persistent, Contextual, Scratchpad).
4. Cognitive Active Forgetting Engine (CAFE) with Orthogonal Subspace Nullification (P_⊥ = I - VV^T).
5. Zero-Token Latent RSSM Rollout Planner (MCTS Tree Search in discrete latent space).
6. SwiGLU 4096 Feed-Forward Network with Pre-RMSNorm and Residual Skips.
"""

import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, PathPatch, Circle
from matplotlib.path import Path

# Configure Matplotlib fonts and rendering quality
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'dejavusans'

def create_rounded_box(ax, x, y, w, h, bg_color, border_color, text, subtext=None, fontsize=10.5, sub_fontsize=7.8, lw=1.6, text_color='#000000', rad=0.08, zorder=3):
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={rad}",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=lw,
        zorder=zorder
    )
    ax.add_patch(box)
    
    if subtext:
        ax.text(x, y + h*0.22, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color=text_color, zorder=zorder+1)
        ax.text(x, y - h*0.22, subtext, ha='center', va='center', fontsize=sub_fontsize, fontweight='normal', color='#1E293B' if text_color == '#000000' else '#F1F5F9', zorder=zorder+1)
    else:
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color=text_color, zorder=zorder+1)
    return box

def draw_straight_arrow(ax, x1, y1, x2, y2, color='#000000', lw=1.6, zorder=2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            mutation_scale=14,
            shrinkA=0,
            shrinkB=0
        ),
        zorder=zorder
    )

def draw_classic_skip(ax, trunk_x, start_y, target_box_right_x, target_box_y, branch_x, color='#000000', lw=1.6, zorder=2):
    """
    Draws a faithful Vaswani-style residual skip connection:
    Branches right from trunk at start_y, runs up along branch_x, turns left into target box right side.
    """
    path_data = [
        (Path.MOVETO, (trunk_x, start_y)),
        (Path.LINETO, (branch_x, start_y)),
        (Path.LINETO, (branch_x, target_box_y)),
        (Path.LINETO, (target_box_right_x, target_box_y))
    ]
    codes, verts = zip(*path_data)
    path = Path(verts, codes)
    patch = PathPatch(path, facecolor='none', edgecolor=color, lw=lw, zorder=zorder)
    ax.add_patch(patch)
    
    ax.annotate(
        "", xy=(target_box_right_x, target_box_y), xytext=(target_box_right_x + 0.15, target_box_y),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            mutation_scale=13,
            shrinkA=0,
            shrinkB=0
        ),
        zorder=zorder
    )

def render_true_architecture():
    """Generates the Comprehensive Dual-Panel Architecture Figure (Full Pipeline + RSSM Deep Dive)."""
    fig, ax = plt.subplots(figsize=(17.0, 15.8), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 17.0)
    ax.set_ylim(0, 15.8)
    ax.axis('off')

    c_bg_embed = '#FCE7F3'     # Soft Rose / Pink
    c_bd_embed = '#000000'
    c_bg_norm  = '#FEF08A'     # Soft Yellow
    c_bd_norm  = '#000000'
    c_bg_ssm   = '#FFEDD5'     # Soft Peach / Orange
    c_bd_ssm   = '#000000'
    c_bg_ffn   = '#BAE6FD'     # Soft Sky Blue
    c_bd_ffn   = '#000000'
    c_bg_prior = '#DDD6FE'     # Soft Violet
    c_bd_prior = '#000000'
    c_bg_head  = '#E0E7FF'     # Soft Indigo
    c_bd_head  = '#000000'
    c_bg_cafe  = '#D1FAE5'     # Soft Mint / Emerald
    c_bd_cafe  = '#000000'
    c_bg_plan  = '#EDE9FE'     # Soft Purple
    c_bd_plan  = '#7C3AED'

    # =========================================================================
    # LEFT PANEL: Full Foundation Pipeline (Inputs -> N x Stack -> Logits)
    # =========================================================================
    cx1 = 4.2

    ax.text(cx1, 15.2, "Hokie-LM / NeuroWorld-LM", ha='center', va='center', fontsize=18, fontweight='bold', color='#000000')
    ax.text(cx1, 14.82, "End-to-End Cognitive World-Model Architecture (Attention-Free, Softmax-Free)", ha='center', va='center', fontsize=9.8, color='#475569')

    # Top: Raw Next-Token Logits Output (No Softmax in Model forward pass!)
    ax.text(cx1, 14.05, r"Output Next-Token Logits $\hat{y}_t \in \mathbb{R}^{V}$", ha='center', va='center', fontsize=12.2, fontweight='bold', color='#000000')
    ax.text(cx1, 13.68, "(Direct Cross-Entropy Loss / Top-p Sampling)", ha='center', va='center', fontsize=8.4, color='#64748B')
    draw_straight_arrow(ax, cx1, 12.95, cx1, 13.45)

    # Linear LM Head (Tied weights)
    create_rounded_box(ax, cx1, 12.65, 3.4, 0.58, c_bg_head, c_bd_head, "Linear (LM Head)", r"Tied Token Weights ($d_{\mathrm{model}} \to V=50,266$)", fontsize=10.5, sub_fontsize=7.8)
    draw_straight_arrow(ax, cx1, 11.65, cx1, 12.36)

    # Final RMSNorm
    create_rounded_box(ax, cx1, 11.38, 3.4, 0.48, c_bg_norm, c_bd_norm, "Final RMSNorm", fontsize=10.5)
    draw_straight_arrow(ax, cx1, 10.55, cx1, 11.14)

    # -------------------------------------------------------------
    # N x Bounded Box (24 Layers)
    # -------------------------------------------------------------
    stack_box = FancyBboxPatch(
        (cx1 - 2.45, 3.00), 4.9, 7.50,
        boxstyle="round,pad=0.0,rounding_size=0.18",
        facecolor='#F8FAFC',
        edgecolor='#000000',
        linewidth=2.2,
        zorder=1
    )
    ax.add_patch(stack_box)

    # Nx Label
    ax.text(cx1 + 3.15, 6.95, r"$N\times$", ha='center', va='center', fontsize=22, fontweight='bold', color='#000000')
    ax.text(cx1 + 3.15, 6.45, "(24 Layers)", ha='center', va='center', fontsize=9.2, fontweight='bold', color='#475569')

    # Inside N x Box:
    # 4. Top Add & RMSNorm
    create_rounded_box(ax, cx1, 9.95, 3.3, 0.48, c_bg_norm, c_bd_norm, "Add & RMSNorm", fontsize=11)
    draw_straight_arrow(ax, cx1, 9.10, cx1, 9.71)

    # 3. SwiGLU Feed Forward
    create_rounded_box(ax, cx1, 8.55, 3.3, 0.92, c_bg_ffn, c_bd_ffn, "SwiGLU Feed Forward", r"$\mathrm{FFN}(x) = W_3(\mathrm{SiLU}(W_1 x) \odot W_2 x)$ (Dim: 4096)", fontsize=11, sub_fontsize=8.0)
    draw_straight_arrow(ax, cx1, 7.58, cx1, 8.09)

    # Skip Connection around FFN (Vaswani style into side of Add & RMSNorm)
    draw_classic_skip(ax, cx1, 7.82, cx1 + 1.65, 9.95, cx1 + 2.05, lw=1.6)

    # 2. Middle Add & RMSNorm
    create_rounded_box(ax, cx1, 7.28, 3.3, 0.48, c_bg_norm, c_bd_norm, "Add & RMSNorm", fontsize=11)
    draw_straight_arrow(ax, cx1, 6.35, cx1, 7.04)

    # 1. RSSM World-Model Layer (SSM + Latent Prior + CAFE)
    ssm_outer = FancyBboxPatch(
        (cx1 - 1.65, 3.75), 3.3, 2.50,
        boxstyle="round,pad=0.0,rounding_size=0.08",
        facecolor=c_bg_ssm,
        edgecolor=c_bd_ssm,
        linewidth=1.8,
        zorder=3
    )
    ax.add_patch(ssm_outer)
    ax.text(cx1, 5.98, "Cognitive RSSM Block", ha='center', va='center', fontsize=11.2, fontweight='bold', color='#000000', zorder=4)

    # Inner Badges
    ssm_sub1 = FancyBboxPatch((cx1 - 1.45, 5.15), 2.9, 0.48, boxstyle="round,pad=0.0,rounding_size=0.05",
                              facecolor='#FEF3C7', edgecolor='#D97706', linewidth=1.0, zorder=4)
    ax.add_patch(ssm_sub1)
    ax.text(cx1, 5.39, r"Categorical Latent Prior: $z_t \sim q(z_t|h_t)$", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#78350F', zorder=5)

    ssm_sub2 = FancyBboxPatch((cx1 - 1.45, 3.90), 2.9, 1.10, boxstyle="round,pad=0.0,rounding_size=0.05",
                              facecolor='#ECFDF5', edgecolor='#059669', linewidth=1.0, zorder=4)
    ax.add_patch(ssm_sub2)
    ax.text(cx1, 4.62, r"Continuous SSM: $h_t = \bar{\mathbf{A}}h_{t-1} + \bar{\mathbf{B}}\tilde{x}_t$", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#065F46', zorder=5)
    ax.text(cx1, 4.22, r"CAFE Subspace Nullification: $\mathbf{P}_\perp = \mathbf{I} - \mathbf{V}\mathbf{V}^T$", ha='center', va='center', fontsize=7.6, fontweight='bold', color='#047857', zorder=5)

    draw_straight_arrow(ax, cx1, 2.70, cx1, 3.75)

    # Skip Connection around SSM (Vaswani style into side of Add & RMSNorm, starting inside stack box)
    draw_classic_skip(ax, cx1, 3.25, cx1 + 1.65, 7.28, cx1 + 2.05, lw=1.6)

    # Bottom: Continuous Time Parameter & Input Embedding
    circle_dt = Circle((cx1 - 1.9, 2.25), 0.32, facecolor='#FFFFFF', edgecolor='#000000', linewidth=1.8, zorder=3)
    ax.add_patch(circle_dt)
    ax.text(cx1 - 1.9, 2.25, r"$\Delta t$", ha='center', va='center', fontsize=12.5, fontweight='bold', color='#000000', zorder=4)
    ax.text(cx1 - 1.9, 1.70, "Continuous\nODE Dynamics", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#000000')

    # Summing Circle (+)
    circle_sum = Circle((cx1, 2.25), 0.22, facecolor='#FFFFFF', edgecolor='#000000', linewidth=1.8, zorder=3)
    ax.add_patch(circle_sum)
    ax.text(cx1, 2.25, "+", ha='center', va='center', fontsize=14, fontweight='bold', color='#000000', zorder=4)

    draw_straight_arrow(ax, cx1 - 1.58, 2.25, cx1 - 0.22, 2.25)
    draw_straight_arrow(ax, cx1, 2.47, cx1, 3.00)
    draw_straight_arrow(ax, cx1, 1.50, cx1, 2.03)

    # Input Embedding
    create_rounded_box(ax, cx1, 1.25, 3.4, 0.48, c_bg_embed, c_bd_embed, "Token Embedding", r"No Positional Embedding ($d_{\mathrm{model}} = 1024$)", fontsize=10.5, sub_fontsize=7.8)
    draw_straight_arrow(ax, cx1, 0.70, cx1, 1.01)

    # Inputs Text
    ax.text(cx1, 0.52, r"Inputs $(x_1, x_2, \dots, x_t)$", ha='center', va='center', fontsize=12, fontweight='bold', color='#000000')


    # =========================================================================
    # RIGHT PANEL: True Internal RSSM, CAFE & Latent Planner
    # =========================================================================
    cx2 = 12.1

    ax.text(cx2, 15.2, "Inside the RSSM Cognitive Cell", ha='center', va='center', fontsize=18, fontweight='bold', color='#000000')
    ax.text(cx2, 14.82, "Multi-Scale Memory, Orthogonal Subspace Eviction & Latent MCTS Planning", ha='center', va='center', fontsize=9.8, color='#475569')

    # Outer Container Box
    zoom_box = FancyBboxPatch(
        (cx2 - 3.55, 0.45), 7.1, 14.1,
        boxstyle="round,pad=0.0,rounding_size=0.18",
        facecolor='#F8FAFC',
        edgecolor='#000000',
        linewidth=2.2,
        zorder=1
    )
    ax.add_patch(zoom_box)

    # 1. Zero-Token Latent Rollout Planner (Top of Right Panel)
    planner_box = FancyBboxPatch((cx2 - 3.30, 11.90), 6.6, 2.35, boxstyle="round,pad=0.0,rounding_size=0.12",
                                 facecolor=c_bg_plan, edgecolor=c_bd_plan, linewidth=2.0, zorder=2)
    ax.add_patch(planner_box)
    
    # Title Header Badge
    create_rounded_box(ax, cx2, 13.90, 6.2, 0.44, '#6D28D9', '#4C1D95', "Zero-Token Latent MCTS / Rollout Planner", fontsize=10.2, text_color='#FFFFFF')
    ax.text(cx2, 13.44, r"Imaginary Forward Rollouts in Discrete Categorical Space: $h_{\tau} = \mathrm{SSM}(h_{\tau-1}, z_{\tau})$", ha='center', va='center', fontsize=8.2, fontweight='bold', color='#4C1D95', zorder=4)

    # Planner 3 Branch Bubbles
    for idx, (bx, label) in enumerate([(-2.0, r"Branch 1 ($\pi_1$)"), (0.0, r"Branch 2 ($\pi_2$)"), (2.0, r"Branch 3 ($\pi_3$)")]):
        b_box = FancyBboxPatch((cx2 + bx - 0.90, 12.10), 1.80, 0.95, boxstyle="round,pad=0.0,rounding_size=0.06",
                               facecolor='#FFFFFF', edgecolor='#7C3AED', linewidth=1.2, zorder=3)
        ax.add_patch(b_box)
        ax.text(cx2 + bx, 12.72, label, ha='center', va='center', fontsize=8.2, fontweight='bold', color='#5B21B6', zorder=4)
        ax.text(cx2 + bx, 12.35, r"$V(h_\tau) = \gamma^\tau r_\tau$", ha='center', va='center', fontsize=7.6, color='#334155', zorder=4)

    draw_straight_arrow(ax, cx2, 11.15, cx2, 11.90, color='#7C3AED')

    # 2. Gated SSM Output & State Projection
    gate_box = FancyBboxPatch((cx2 - 3.10, 10.05), 6.2, 1.10, boxstyle="round,pad=0.0,rounding_size=0.08",
                              facecolor='#FEF08A', edgecolor='#000000', linewidth=1.5, zorder=3)
    ax.add_patch(gate_box)
    ax.text(cx2, 10.78, r"Gated SSM Output: $y_t = ( \sum_{s=1}^{16} h_{t,s} C_{t,s} + D x_t ) \odot \mathrm{SiLU}(z_t)$",
            ha='center', va='center', fontsize=9.4, fontweight='bold', color='#000000', zorder=4)
    ax.text(cx2, 10.36, r"Multiplicative Gating branch $z_t = \mathrm{Linear}(u_t)$ regulates information flow",
            ha='center', va='center', fontsize=8.2, color='#475569', zorder=4)

    draw_straight_arrow(ax, cx2, 9.45, cx2, 10.05)

    # 3. Multi-Scale Working Memory & CAFE Engine (Centerpiece)
    cafe_box = FancyBboxPatch((cx2 - 3.30, 4.30), 6.6, 5.15, boxstyle="round,pad=0.0,rounding_size=0.12",
                              facecolor='#ECFDF5', edgecolor='#059669', linewidth=2.0, zorder=2)
    ax.add_patch(cafe_box)

    # Title Header Badge
    create_rounded_box(ax, cx2, 9.15, 6.2, 0.44, '#059669', '#047857', "Cognitive Active Forgetting Engine (CAFE) & 17 KB Memory", fontsize=9.4, text_color='#FFFFFF')

    # Left: Multi-Scale State Buffers
    ms_box = FancyBboxPatch((cx2 - 3.05, 5.65), 2.90, 2.95, boxstyle="round,pad=0.0,rounding_size=0.06",
                            facecolor=c_bg_ssm, edgecolor=c_bd_ssm, linewidth=1.4, zorder=3)
    ax.add_patch(ms_box)
    ax.text(cx2 - 1.60, 8.25, "Multi-Scale State Partition", ha='center', va='center', fontsize=9.2, fontweight='bold', color='#000000', zorder=4)
    ax.text(cx2 - 1.60, 7.72, "• Persistent Memory ($h_{\\mathrm{perm}}$)", ha='center', va='center', fontsize=8.0, color='#065F46', zorder=4)
    ax.text(cx2 - 1.60, 7.22, "• Contextual Flow ($h_{\\mathrm{ctx}}$)", ha='center', va='center', fontsize=8.0, color='#065F46', zorder=4)
    ax.text(cx2 - 1.60, 6.72, "• Scratchpad Registers ($h_{\\mathrm{scr}}$)", ha='center', va='center', fontsize=8.0, color='#991B1B', zorder=4)
    ax.text(cx2 - 1.60, 6.12, r"$h_t = \bar{\mathbf{A}}h_{t-1} + \bar{\mathbf{B}}\tilde{x}_t$  (17 KB / Layer)", ha='center', va='center', fontsize=7.6, fontweight='bold', color='#000000', zorder=4)

    # Right: CAFE Subspace Eviction & Scratchpad Clearing
    evict_box = FancyBboxPatch((cx2 + 0.15, 5.65), 2.90, 2.95, boxstyle="round,pad=0.0,rounding_size=0.06",
                               facecolor=c_bg_cafe, edgecolor=c_bd_cafe, linewidth=1.4, zorder=3)
    ax.add_patch(evict_box)
    ax.text(cx2 + 1.60, 8.25, "Active Forgetting Operators", ha='center', va='center', fontsize=9.2, fontweight='bold', color='#065F46', zorder=4)
    ax.text(cx2 + 1.60, 7.62, r"$\mathbf{P}_\perp = \mathbf{I} - \mathbf{V}(\mathbf{V}^T \mathbf{V})^{-1}\mathbf{V}^T$", ha='center', va='center', fontsize=8.2, fontweight='bold', color='#065F46', zorder=4)
    ax.text(cx2 + 1.60, 7.12, r"Subspace Nullification: $h_t \leftarrow \mathbf{P}_\perp h_t$", ha='center', va='center', fontsize=7.8, color='#047857', zorder=4)
    ax.text(cx2 + 1.60, 6.64, r"Scratchpad Evict: $h_{\mathrm{scr}} \leftarrow 0$", ha='center', va='center', fontsize=7.8, fontweight='bold', color='#991B1B', zorder=4)
    ax.text(cx2 + 1.60, 6.12, r"$I(\mathrm{Target}; h_t) \equiv 0$ (100% Evict)", ha='center', va='center', fontsize=7.6, fontweight='bold', color='#065F46', zorder=4)

    # Bidirectional communication arrow
    ax.annotate("", xy=(cx2 + 0.15, 7.12), xytext=(cx2 - 0.15, 7.12),
                arrowprops=dict(arrowstyle="<->", color='#059669', lw=2.0, mutation_scale=13), zorder=5)

    ax.text(cx2, 4.88, r"Selective SSM Matrices: $\bar{\mathbf{A}} = \exp(-\Delta t \cdot \mathbf{A}), \;\; \bar{\mathbf{B}} = \Delta t \cdot \mathbf{B}(\tilde{x}_t), \;\; \mathbf{C} = \mathbf{C}(\tilde{x}_t)$",
            ha='center', va='center', fontsize=8.6, fontweight='bold', color='#065F46', zorder=4)

    draw_straight_arrow(ax, cx2, 3.75, cx2, 4.30)

    # 4. Dual-Stream Categorical Latent World Model (Prior & Posterior)
    prior_box = FancyBboxPatch((cx2 - 3.30, 1.35), 6.6, 2.4, boxstyle="round,pad=0.0,rounding_size=0.12",
                               facecolor='#F5F3FF', edgecolor='#7C3AED', linewidth=2.0, zorder=2)
    ax.add_patch(prior_box)

    # Title Header Badge
    create_rounded_box(ax, cx2, 3.48, 6.2, 0.42, '#6D28D9', '#4C1D95', "Dual-Stream Categorical Latent World Model", fontsize=10.2, text_color='#FFFFFF')

    # Prior
    create_rounded_box(ax, cx2 - 1.6, 2.50, 2.85, 1.20, c_bg_prior, c_bd_prior,
                       "Prior Network",
                       r"$p(z_t | h_{t-1})$" + "\n" + r"$\mathrm{MLP}(h_{t-1}) \to 8 \times 8$" + "\n(Autonomous Generation)",
                       fontsize=9.2, sub_fontsize=7.4, rad=0.06)

    # Posterior
    create_rounded_box(ax, cx2 + 1.6, 2.50, 2.85, 1.20, '#EDE9FE', '#7C3AED',
                       "Posterior Network",
                       r"$q(z_t | h_{t-1}, x_t)$" + "\n" + r"$\mathrm{MLP}([h_{t-1}, x_t])$" + "\n(Teacher-Forced Training)",
                       fontsize=9.2, sub_fontsize=7.4, rad=0.06)

    # Surprise Gate KL badge
    sg_box = FancyBboxPatch((cx2 - 2.6, 1.5), 5.2, 0.46, boxstyle="round,pad=0.0,rounding_size=0.04",
                            facecolor='#FFFFFF', edgecolor='#7C3AED', linewidth=1.0, zorder=3)
    ax.add_patch(sg_box)
    ax.text(cx2, 1.73, r"Surprise Gate: $\mathcal{D}_{\mathrm{KL}}(q \parallel p) \to \text{Dynamically modulates SSM state update}$",
            ha='center', va='center', fontsize=8.0, fontweight='bold', color='#5B21B6', zorder=4)

    draw_straight_arrow(ax, cx2, 0.88, cx2, 1.35)

    # Layer Input at bottom of Zoom
    ax.text(cx2, 0.68, r"Input: $x_t^{(l-1)} \in \mathbb{R}^{B \times d_{\mathrm{model}}}$  (Fused Token + Latent Vector $\tilde{x}_t$)",
            ha='center', va='center', fontsize=10.5, fontweight='bold', color='#000000')

    # =========================================================================
    # FOOTER & COMPARISON BANNER
    # =========================================================================
    banner = FancyBboxPatch((0.9, 0.04), 15.2, 0.32, boxstyle="round,pad=0.0,rounding_size=0.05",
                            facecolor='#0F172A', edgecolor='#000000', linewidth=1.2, zorder=2)
    ax.add_patch(banner)
    ax.text(8.5, 0.20,
            "Key Breakthroughs: ① O(1) Constant Working Memory (17 KB)  •  ② Zero Softmax Attention Bottleneck  •  ③ CAFE Privacy Forgetting (P_⊥)  •  ④ Zero-Token Latent MCTS Planning",
            ha='center', va='center', fontsize=8.2, fontweight='bold', color='#F8FAFC', zorder=3)

    out_dir = "/home/eun/neuroworld_lm/figures"
    os.makedirs(out_dir, exist_ok=True)
    
    png_path = os.path.join(out_dir, "hokie_model_architecture.png")
    svg_path = os.path.join(out_dir, "hokie_model_architecture.svg")
    pdf_path = os.path.join(out_dir, "hokie_model_architecture.pdf")

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()

    # Sync to paper
    paper_fig_dir = "/home/eun/neuroworld_lm/paper/figures"
    if os.path.exists(paper_fig_dir):
        shutil.copy2(pdf_path, os.path.join(paper_fig_dir, "fig1_model_architecture.pdf"))

    print(f"[✓] Dual-Panel Architecture Diagram saved to: {png_path}")

def render_classic_single_plate_true():
    """Generates the True Single-Column Plate without Softmax attention/top bottlenecks."""
    fig, ax = plt.subplots(figsize=(8.5, 13.8), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 8.5)
    ax.set_ylim(0, 13.8)
    ax.axis('off')

    c_bg_embed = '#FCE7F3'     # Soft Rose / Pink
    c_bd_embed = '#000000'
    c_bg_norm  = '#FEF08A'     # Soft Yellow
    c_bd_norm  = '#000000'
    c_bg_ssm   = '#FFEDD5'     # Soft Peach / Orange
    c_bd_ssm   = '#000000'
    c_bg_ffn   = '#BAE6FD'     # Soft Sky Blue
    c_bd_ffn   = '#000000'
    c_bg_head  = '#E0E7FF'     # Soft Indigo
    c_bd_head  = '#000000'

    cx = 4.25

    # Top: Next-Token Logits Output (No Softmax block!)
    ax.text(cx, 13.15, "Output Next-Token\nLogits " + r"$\hat{y}_t \in \mathbb{R}^{V}$", ha='center', va='center', fontsize=12.5, fontweight='bold', color='#000000')
    ax.text(cx, 12.60, "(Direct Cross-Entropy Loss / Top-p Sampling)", ha='center', va='center', fontsize=8.4, color='#64748B')
    draw_straight_arrow(ax, cx, 11.85, cx, 12.40)

    # Linear LM Head
    create_rounded_box(ax, cx, 11.52, 3.6, 0.58, c_bg_head, c_bd_head, "Linear (LM Head)", r"Tied Token Weights ($d_{\mathrm{model}} \to 50,266$)", fontsize=10.5, sub_fontsize=7.8)
    draw_straight_arrow(ax, cx, 10.55, cx, 11.23)

    # Final RMSNorm
    create_rounded_box(ax, cx, 10.28, 3.6, 0.48, c_bg_norm, c_bd_norm, "Final RMSNorm", fontsize=10.5)
    draw_straight_arrow(ax, cx, 9.45, cx, 10.04)

    # N x Stack Box
    stack_box = FancyBboxPatch(
        (cx - 2.40, 2.35), 4.8, 6.75,
        boxstyle="round,pad=0.0,rounding_size=0.18",
        facecolor='#FFFFFF',
        edgecolor='#000000',
        linewidth=2.0,
        zorder=1
    )
    ax.add_patch(stack_box)
    
    # Nx Label placed with clear margin to the right
    ax.text(cx + 2.95, 5.85, r"$N\times$", ha='center', va='center', fontsize=20, fontweight='bold', color='#000000')
    ax.text(cx + 2.95, 5.35, "(24 Layers)", ha='center', va='center', fontsize=8.8, fontweight='bold', color='#64748B')

    # Add & RMSNorm (Top)
    create_rounded_box(ax, cx, 8.55, 3.4, 0.48, c_bg_norm, c_bd_norm, "Add & RMSNorm", fontsize=11)
    draw_straight_arrow(ax, cx, 7.72, cx, 8.31)

    # Feed Forward (SwiGLU)
    create_rounded_box(ax, cx, 7.25, 3.4, 0.85, c_bg_ffn, c_bd_ffn, "Feed Forward", r"$\mathrm{SwiGLU}(x) = W_3(\mathrm{SiLU}(W_1 x) \odot W_2 x)$" + "\n(4096-dim / Latent Rollout Planner)", fontsize=10.8, sub_fontsize=7.8)
    draw_straight_arrow(ax, cx, 6.28, cx, 6.82)

    # Skip 2 around FFN (into side of top Add & RMSNorm)
    draw_classic_skip(ax, cx, 6.52, cx + 1.70, 8.55, cx + 2.10, lw=1.6)

    # Add & RMSNorm (Middle)
    create_rounded_box(ax, cx, 6.02, 3.4, 0.48, c_bg_norm, c_bd_norm, "Add & RMSNorm", fontsize=11)
    draw_straight_arrow(ax, cx, 5.15, cx, 5.78)

    # Exact Selective SSM Core with Latent Prior & CAFE
    ssm_box = FancyBboxPatch(
        (cx - 1.70, 2.80), 3.4, 2.30,
        boxstyle="round,pad=0.0,rounding_size=0.08",
        facecolor=c_bg_ssm,
        edgecolor=c_bd_ssm,
        linewidth=1.8,
        zorder=3
    )
    ax.add_patch(ssm_box)
    ax.text(cx, 4.85, "Cognitive RSSM Block", ha='center', va='center', fontsize=11.2, fontweight='bold', color='#000000', zorder=4)
    
    # Internal sub-boxes
    ssm_sub1 = FancyBboxPatch((cx - 1.50, 4.10), 3.0, 0.45, boxstyle="round,pad=0.0,rounding_size=0.04",
                              facecolor='#FEF3C7', edgecolor='#D97706', linewidth=0.9, zorder=4)
    ax.add_patch(ssm_sub1)
    ax.text(cx, 4.32, r"Categorical Prior: $z_t \sim q(z_t|h_t)$", ha='center', va='center', fontsize=7.8, fontweight='bold', color='#78350F', zorder=5)

    ssm_sub2 = FancyBboxPatch((cx - 1.50, 2.95), 3.0, 1.02, boxstyle="round,pad=0.0,rounding_size=0.04",
                              facecolor='#ECFDF5', edgecolor='#059669', linewidth=0.9, zorder=4)
    ax.add_patch(ssm_sub2)
    ax.text(cx, 3.62, r"Continuous SSM: $h_t = \bar{\mathbf{A}}h_{t-1} + \bar{\mathbf{B}}\tilde{x}_t$", ha='center', va='center', fontsize=7.8, fontweight='bold', color='#065F46', zorder=5)
    ax.text(cx, 3.25, r"CAFE Forgetting: $\mathbf{P}_\perp = \mathbf{I} - \mathbf{V}\mathbf{V}^T$", ha='center', va='center', fontsize=7.5, fontweight='bold', color='#047857', zorder=5)

    draw_straight_arrow(ax, cx, 1.95, cx, 2.80)

    # Skip 1 around SSM (into side of middle Add & RMSNorm, starting cleanly inside stack box)
    draw_classic_skip(ax, cx, 2.58, cx + 1.70, 6.02, cx + 2.10, lw=1.6)

    # Delta t Continuous Dynamics
    circle_dt = Circle((cx - 1.85, 1.75), 0.32, facecolor='#FFFFFF', edgecolor='#000000', linewidth=1.6, zorder=3)
    ax.add_patch(circle_dt)
    ax.text(cx - 1.85, 1.75, r"$\Delta t$", ha='center', va='center', fontsize=12, fontweight='bold', color='#000000', zorder=4)
    ax.text(cx - 1.85, 1.20, "Continuous\nODE Dynamics", ha='center', va='center', fontsize=7.8, fontweight='bold', color='#000000')

    # Summing (+)
    circle_sum = Circle((cx, 1.75), 0.22, facecolor='#FFFFFF', edgecolor='#000000', linewidth=1.6, zorder=3)
    ax.add_patch(circle_sum)
    ax.text(cx, 1.75, "+", ha='center', va='center', fontsize=13, fontweight='bold', color='#000000', zorder=4)

    draw_straight_arrow(ax, cx - 1.53, 1.75, cx - 0.22, 1.75)
    draw_straight_arrow(ax, cx, 1.05, cx, 1.53)

    # Input Embedding
    create_rounded_box(ax, cx, 0.82, 3.4, 0.48, c_bg_embed, c_bd_embed, "Token Embedding", r"Vocab: 50,266 | $d_{\mathrm{model}} = 1024$", fontsize=10.5, sub_fontsize=7.8)
    draw_straight_arrow(ax, cx, 0.28, cx, 0.58)

    # Inputs
    ax.text(cx, 0.12, r"Inputs $(x_1, x_2, \dots, x_t)$", ha='center', va='center', fontsize=11.5, fontweight='bold', color='#000000')

    out_dir = "/home/eun/neuroworld_lm/figures"
    plate_path = os.path.join(out_dir, "hokie_architecture_classic_plate.png")
    plt.savefig(plate_path, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, "hokie_architecture_classic_plate.pdf"), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, "hokie_architecture_classic_plate.svg"), format='svg', bbox_inches='tight')
    plt.close()
    print(f"[✓] Classic True Plate Diagram saved to: {plate_path}")

if __name__ == "__main__":
    render_true_architecture()
    render_classic_single_plate_true()
