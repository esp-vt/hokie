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

def draw_autoregressive_loop(ax, start_x, start_y, target_x, target_y, route_x, color='#DC2626', lw=1.8, label="Autoregressive Loop (t -> t+1)"):
    """
    Draws a clean, prominent loop line from top output down to bottom input.
    """
    path_data = [
        (Path.MOVETO, (start_x, start_y)),
        (Path.LINETO, (route_x, start_y)),
        (Path.LINETO, (route_x, target_y)),
        (Path.LINETO, (target_x, target_y))
    ]
    codes, verts = zip(*path_data)
    path = Path(verts, codes)
    patch = PathPatch(path, facecolor='none', edgecolor=color, lw=lw, linestyle='--', zorder=3)
    ax.add_patch(patch)
    
    ax.annotate(
        "", xy=(target_x, target_y), xytext=(target_x - 0.15, target_y),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            mutation_scale=14,
            shrinkA=0,
            shrinkB=0
        ),
        zorder=4
    )
    # Midpoint label rotated along the line
    mid_y = (start_y + target_y) / 2.0
    ax.text(route_x - 0.22, mid_y, label, ha='center', va='center', rotation=90,
            fontsize=8.5, fontweight='bold', color=color, zorder=5)

def draw_classic_skip(ax, start_x, start_y, end_x, end_y, route_x, color='#000000', lw=1.6):
    """
    Draws a standard residual skip connection around a sub-block.
    """
    path_data = [
        (Path.MOVETO, (start_x, start_y)),
        (Path.LINETO, (route_x, start_y)),
        (Path.LINETO, (route_x, end_y)),
        (Path.LINETO, (end_x, end_y))
    ]
    codes, verts = zip(*path_data)
    path = Path(verts, codes)
    patch = PathPatch(path, facecolor='none', edgecolor=color, lw=lw, zorder=2)
    ax.add_patch(patch)
    
    # Circle tap at start
    circle = Circle((start_x, start_y), 0.04, facecolor=color, edgecolor=color, zorder=4)
    ax.add_patch(circle)
    
    # Arrow into end
    ax.annotate(
        "", xy=(end_x, end_y), xytext=(end_x + 0.12, end_y),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            mutation_scale=12,
            shrinkA=0,
            shrinkB=0
        ),
        zorder=4
    )



def render_true_architecture():
    """Generates the Comprehensive Dual-Panel Architecture Figure with Autoregressive Loop."""
    fig, ax = plt.subplots(figsize=(17.0, 15.0), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 17.0)
    ax.set_ylim(0, 15.0)
    ax.axis('off')

    c_bg_embed = '#FCE7F3'     # Soft Rose
    c_bd_embed = '#000000'
    c_bg_norm  = '#FEF08A'     # Soft Yellow
    c_bd_norm  = '#000000'
    c_bg_ssm   = '#FFEDD5'     # Soft Peach
    c_bd_ssm   = '#000000'
    c_bg_ffn   = '#BAE6FD'     # Soft Sky Blue
    c_bd_ffn   = '#000000'
    c_bg_prior = '#DDD6FE'     # Soft Violet
    c_bd_prior = '#000000'
    c_bg_head  = '#E0E7FF'     # Soft Indigo
    c_bd_head  = '#000000'
    c_bg_cafe  = '#D1FAE5'     # Soft Mint
    c_bd_cafe  = '#000000'

    # =========================================================================
    # LEFT PANEL: Full Foundation Pipeline (Inputs -> N x Stack -> Logits)
    # =========================================================================
    cx1 = 4.6

    ax.text(cx1, 14.50, "Hokie-LM Architecture", ha='center', va='center', fontsize=18, fontweight='bold', color='#0F172A')
    ax.text(cx1, 14.15, "24-Layer Cognitive World Model", ha='center', va='center', fontsize=10, color='#64748B')

    # Top: Next-Token Output Box
    create_rounded_box(ax, cx1, 13.35, 3.4, 0.55, '#F1F5F9', '#DC2626', r"Output Token $\hat{x}_{t+1}$", r"(Next-Token Logits $\hat{y}_t \in \mathbb{R}^V$)", fontsize=10.5, sub_fontsize=7.8, text_color='#DC2626', lw=1.8)
    draw_straight_arrow(ax, cx1, 12.65, cx1, 13.07)

    # Linear LM Head
    create_rounded_box(ax, cx1, 12.35, 3.4, 0.52, c_bg_head, c_bd_head, "Linear LM Head", r"Tied Weights ($d_{\mathrm{model}} \to V$)", fontsize=10.5, sub_fontsize=7.8)
    draw_straight_arrow(ax, cx1, 11.50, cx1, 12.09)

    # Final RMSNorm
    create_rounded_box(ax, cx1, 11.25, 3.4, 0.46, c_bg_norm, c_bd_norm, "Final RMSNorm", fontsize=10.5)
    draw_straight_arrow(ax, cx1, 10.45, cx1, 11.02)

    # -------------------------------------------------------------
    # N x Bounded Box (24 Layers)
    # -------------------------------------------------------------
    stack_box = FancyBboxPatch(
        (cx1 - 2.40, 2.65), 4.8, 7.80,
        boxstyle="round,pad=0.0,rounding_size=0.18",
        facecolor='#F8FAFC',
        edgecolor='#0F172A',
        linewidth=2.0,
        zorder=1
    )
    ax.add_patch(stack_box)

    # Nx Label
    ax.text(cx1 + 3.05, 6.75, r"$N\times$", ha='center', va='center', fontsize=22, fontweight='bold', color='#0F172A')
    ax.text(cx1 + 3.05, 6.25, "(24 Layers)", ha='center', va='center', fontsize=9.2, fontweight='bold', color='#64748B')

    # Inside N x Box:
    # 4. Top Add & RMSNorm
    create_rounded_box(ax, cx1, 9.85, 3.3, 0.46, c_bg_norm, c_bd_norm, "Add & RMSNorm", fontsize=10.8)
    draw_straight_arrow(ax, cx1, 9.05, cx1, 9.62)

    # 3. SwiGLU Feed Forward
    create_rounded_box(ax, cx1, 8.45, 3.3, 0.85, c_bg_ffn, c_bd_ffn, "SwiGLU FFN", r"$\mathrm{FFN}(x) = W_3(\mathrm{SiLU}(W_1 x) \odot W_2 x)$ (4096-dim)", fontsize=10.8, sub_fontsize=7.8)
    draw_straight_arrow(ax, cx1, 7.45, cx1, 8.02)

    # Skip Connection around FFN
    draw_classic_skip(ax, cx1, 7.70, cx1 + 1.65, 9.85, cx1 + 2.05, lw=1.6)

    # 2. Middle Add & RMSNorm
    create_rounded_box(ax, cx1, 7.18, 3.3, 0.46, c_bg_norm, c_bd_norm, "Add & RMSNorm", fontsize=10.8)
    draw_straight_arrow(ax, cx1, 6.25, cx1, 6.95)

    # 1. Cognitive RSSM Block
    ssm_outer = FancyBboxPatch(
        (cx1 - 1.65, 3.55), 3.3, 2.70,
        boxstyle="round,pad=0.0,rounding_size=0.08",
        facecolor=c_bg_ssm,
        edgecolor=c_bd_ssm,
        linewidth=1.8,
        zorder=3
    )
    ax.add_patch(ssm_outer)
    ax.text(cx1, 5.95, "Cognitive RSSM Block", ha='center', va='center', fontsize=11.2, fontweight='bold', color='#000000', zorder=4)

    # Sub-box 1: CAFE Subspace Engine
    ssm_sub1 = FancyBboxPatch((cx1 - 1.45, 4.80), 2.9, 0.90, boxstyle="round,pad=0.0,rounding_size=0.05",
                              facecolor='#ECFDF5', edgecolor='#059669', linewidth=1.1, zorder=4)
    ax.add_patch(ssm_sub1)
    ax.text(cx1, 5.38, "CAFE Active Forgetting", ha='center', va='center', fontsize=8.8, fontweight='bold', color='#065F46', zorder=5)
    ax.text(cx1, 5.02, r"$\mathbf{P}_\perp = \mathbf{I} - \mathbf{V}\mathbf{V}^T, \;\; h_t = \bar{\mathbf{A}}h_{t-1} + \bar{\mathbf{B}}\tilde{x}_t$", ha='center', va='center', fontsize=7.8, color='#047857', zorder=5)

    # Sub-box 2: Rollout Planner
    ssm_sub2 = FancyBboxPatch((cx1 - 1.45, 3.75), 2.9, 0.85, boxstyle="round,pad=0.0,rounding_size=0.05",
                              facecolor='#F5F3FF', edgecolor='#7C3AED', linewidth=1.1, zorder=4)
    ax.add_patch(ssm_sub2)
    ax.text(cx1, 4.30, "Latent Rollout Planner", ha='center', va='center', fontsize=8.8, fontweight='bold', color='#5B21B6', zorder=5)
    ax.text(cx1, 3.96, r"Zero-Token MCTS: $h_\tau = \mathrm{SSM}(h_{\tau-1}, z_\tau)$", ha='center', va='center', fontsize=7.8, color='#4C1D95', zorder=5)

    draw_straight_arrow(ax, cx1, 2.30, cx1, 3.55)

    # Skip Connection around SSM
    draw_classic_skip(ax, cx1, 2.95, cx1 + 1.65, 7.18, cx1 + 2.05, lw=1.6)

    # Direct Embedding Trunk
    draw_straight_arrow(ax, cx1, 1.45, cx1, 2.30)

    # Input Embedding
    create_rounded_box(ax, cx1, 1.15, 3.4, 0.55, c_bg_embed, c_bd_embed, "Token Embedding", r"Direct Flow ($d_{\mathrm{model}} = 1024$)", fontsize=10.5, sub_fontsize=7.6)
    draw_straight_arrow(ax, cx1, 0.55, cx1, 0.87)

    # Inputs Text Box
    create_rounded_box(ax, cx1, 0.35, 3.4, 0.45, '#F8FAFC', '#0F172A', r"Input Token $x_t$", fontsize=10.8)

    # Autoregressive Loop from Top Output to Bottom Input
    draw_autoregressive_loop(ax, cx1 - 1.70, 13.35, cx1 - 1.70, 0.35, route_x=1.10, color='#DC2626', lw=1.8, label="Autoregressive Feedback Loop (t -> t+1)")


    # =========================================================================
    # RIGHT PANEL: Inside Cognitive RSSM Cell Deep Dive
    # =========================================================================
    cx2 = 12.3

    ax.text(cx2, 14.50, "Inside the Cognitive RSSM Cell", ha='center', va='center', fontsize=17, fontweight='bold', color='#0F172A')
    ax.text(cx2, 14.15, "Internal 5-Stage Dataflow", ha='center', va='center', fontsize=10, color='#64748B')

    # Outer Container Box
    zoom_box = FancyBboxPatch(
        (cx2 - 3.45, 0.15), 6.9, 13.7,
        boxstyle="round,pad=0.0,rounding_size=0.18",
        facecolor='#F8FAFC',
        edgecolor='#0F172A',
        linewidth=2.0,
        zorder=1
    )
    ax.add_patch(zoom_box)

    # 1. Top: Latent Rollout Planner
    create_rounded_box(ax, cx2, 12.45, 6.2, 1.80, '#F5F3FF', '#7C3AED',
                       "5. Latent Rollout Planner",
                       r"Zero-Token MCTS Search in Latent Space: $h_\tau = \mathrm{SSM}(h_{\tau-1}, z_\tau)$" + "\n" + r"Multi-branch Rollouts ($V(h_\tau) = \sum \gamma^\tau r_\tau$)",
                       fontsize=10.2, sub_fontsize=8.0, lw=1.5, rad=0.06)
    draw_straight_arrow(ax, cx2, 11.00, cx2, 11.55, color='#7C3AED')

    # 2. Gated SSM Output
    create_rounded_box(ax, cx2, 10.35, 6.2, 1.05, '#FEF08A', '#000000',
                       "4. Gated SSM Output",
                       r"$y_t = ( \mathbf{C}_t h_t + \mathbf{D} x_t ) \odot \mathrm{SiLU}(z_t)$" + "\n" + r"Emits refined feature vector to Add & RMSNorm",
                       fontsize=10.0, sub_fontsize=8.0, lw=1.4, rad=0.06)
    draw_straight_arrow(ax, cx2, 9.15, cx2, 9.82)

    # 3. Multi-Scale Memory & CAFE Engine
    create_rounded_box(ax, cx2, 7.30, 6.2, 3.30, '#ECFDF5', '#059669',
                       "3. Multi-Scale State Update & CAFE Active Forgetting",
                       r"• 17 KB State: $h_t = \bar{\mathbf{A}}(\Delta t) h_{t-1} + \bar{\mathbf{B}}(\Delta t) \tilde{x}_t$" + "\n" +
                       r"• Persistent ($h_{\mathrm{perm}}$), Contextual ($h_{\mathrm{ctx}}$), Scratchpad ($h_{\mathrm{scr}}$)" + "\n" +
                       r"• CAFE Subspace Nullification: $\mathbf{P}_\perp = \mathbf{I} - \mathbf{V}\mathbf{V}^T, \;\; h_t \leftarrow \mathbf{P}_\perp h_t$" + "\n" +
                       r"• Continuous ODE: $\bar{\mathbf{A}} = \exp(-\Delta t \mathbf{A}), \;\; \bar{\mathbf{B}} = \Delta t \mathbf{B}(\tilde{x}_t)$",
                       fontsize=10.0, sub_fontsize=8.0, lw=1.6, rad=0.06)
    draw_straight_arrow(ax, cx2, 5.00, cx2, 5.65)

    # 4. Input Fusion
    create_rounded_box(ax, cx2, 4.35, 6.2, 0.95, '#F1F5F9', '#475569',
                       "2. Input Fusion & Continuous ODE",
                       r"$\tilde{x}_t = \mathrm{Linear}([x_t, z_t]), \;\; \Delta t = \mathrm{softplus}(\mathbf{W}_\Delta \tilde{x}_t)$",
                       fontsize=9.8, sub_fontsize=8.2, lw=1.3, rad=0.06)
    draw_straight_arrow(ax, cx2, 3.25, cx2, 3.87)

    # 5. Dual-Stream Categorical Latents
    create_rounded_box(ax, cx2, 1.85, 6.2, 2.35, '#F5F3FF', '#7C3AED',
                       "1. Dual-Stream Categorical Latent World Model",
                       r"• Prior Network: $p(z_t | h_{t-1}) = \mathrm{MLP}(h_{t-1})$" + "\n" +
                       r"• Posterior Network: $q(z_t | h_{t-1}, x_t) = \mathrm{MLP}([h_{t-1}, x_t])$" + "\n" +
                       r"• Surprise Gate: $\gamma_t = \mathcal{D}_{\mathrm{KL}}(q \parallel p)$ (Novelty-driven memory update)",
                       fontsize=10.0, sub_fontsize=8.0, lw=1.6, rad=0.06)
    draw_straight_arrow(ax, cx2, 0.40, cx2, 0.67)

    # Bottom Input Label
    ax.text(cx2, 0.28, r"Current Input $x_t$ & Recurrent Memory $h_{t-1}$", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#0F172A')

    out_dir = "/home/eun/neuroworld_lm/figures"
    os.makedirs(out_dir, exist_ok=True)
    
    png_path = os.path.join(out_dir, "hokie_model_architecture.png")
    svg_path = os.path.join(out_dir, "hokie_model_architecture.svg")
    pdf_path = os.path.join(out_dir, "hokie_model_architecture.pdf")

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()

    # Sync to fig1 and paper
    fig1_png = os.path.join(out_dir, "fig1_model_architecture.png")
    fig1_pdf = os.path.join(out_dir, "fig1_model_architecture.pdf")
    shutil.copy2(png_path, fig1_png)
    shutil.copy2(pdf_path, fig1_pdf)

    paper_fig_dir = "/home/eun/neuroworld_lm/paper/figures"
    if os.path.exists(paper_fig_dir):
        shutil.copy2(png_path, os.path.join(paper_fig_dir, "hokie_model_architecture.png"))
        shutil.copy2(pdf_path, os.path.join(paper_fig_dir, "hokie_model_architecture.pdf"))
        shutil.copy2(png_path, os.path.join(paper_fig_dir, "fig1_model_architecture.png"))
        shutil.copy2(pdf_path, os.path.join(paper_fig_dir, "fig1_model_architecture.pdf"))

    print(f"[✓] Streamlined Architecture Diagram with Autoregressive Loop saved to: {png_path}")


def render_classic_single_plate_true():
    """Generates the Single-Column Minimalist Plate exactly matching the clean user diagram."""
    fig, ax = plt.subplots(figsize=(8.0, 13.5), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 8.0)
    ax.set_ylim(0, 13.5)
    ax.axis('off')

    c_bg_embed = '#FCE7F3'
    c_bd_embed = '#000000'
    c_bg_norm  = '#FEF08A'
    c_bd_norm  = '#000000'
    c_bg_ssm   = '#FFEDD5'
    c_bd_ssm   = '#000000'
    c_bg_ffn   = '#BAE6FD'
    c_bd_ffn   = '#000000'
    c_bg_head  = '#E0E7FF'
    c_bd_head  = '#000000'

    cx = 4.4

    # Top: Next Token Output
    create_rounded_box(ax, cx, 12.35, 3.4, 0.55, '#F1F5F9', '#DC2626', r"Output Token $\hat{x}_{t+1}$", fontsize=11, text_color='#DC2626', lw=1.8)
    draw_straight_arrow(ax, cx, 11.60, cx, 12.07)

    # Linear LM Head
    create_rounded_box(ax, cx, 11.35, 3.4, 0.52, c_bg_head, c_bd_head, "Linear LM Head", fontsize=11)
    draw_straight_arrow(ax, cx, 10.50, cx, 11.09)

    # Final RMSNorm
    create_rounded_box(ax, cx, 10.25, 3.4, 0.46, c_bg_norm, c_bd_norm, "Final RMSNorm", fontsize=10.5)
    draw_straight_arrow(ax, cx, 9.40, cx, 10.02)

    # N x Stack Box
    stack_box = FancyBboxPatch(
        (cx - 2.40, 2.25), 4.8, 7.50,
        boxstyle="round,pad=0.0,rounding_size=0.18",
        facecolor='#FFFFFF',
        edgecolor='#000000',
        linewidth=2.0,
        zorder=1
    )
    ax.add_patch(stack_box)
    
    ax.text(cx + 3.05, 6.00, r"$N\times$", ha='center', va='center', fontsize=20, fontweight='bold', color='#000000')

    # Add & RMSNorm (Top)
    create_rounded_box(ax, cx, 8.85, 3.3, 0.46, c_bg_norm, c_bd_norm, "Add & RMSNorm", fontsize=10.8)
    draw_straight_arrow(ax, cx, 8.00, cx, 8.62)

    # Feed Forward (SwiGLU)
    create_rounded_box(ax, cx, 7.45, 3.3, 0.85, c_bg_ffn, c_bd_ffn, "SwiGLU FFN", fontsize=11)
    draw_straight_arrow(ax, cx, 6.45, cx, 7.02)

    # Skip 2 around FFN
    draw_classic_skip(ax, cx, 6.70, cx + 1.65, 8.85, cx + 2.05, lw=1.6)

    # Add & RMSNorm (Middle)
    create_rounded_box(ax, cx, 6.18, 3.3, 0.46, c_bg_norm, c_bd_norm, "Add & RMSNorm", fontsize=10.8)
    draw_straight_arrow(ax, cx, 5.25, cx, 5.95)

    # RSSM Block
    ssm_box = FancyBboxPatch(
        (cx - 1.65, 2.85), 3.3, 2.60,
        boxstyle="round,pad=0.0,rounding_size=0.08",
        facecolor=c_bg_ssm,
        edgecolor=c_bd_ssm,
        linewidth=1.8,
        zorder=3
    )
    ax.add_patch(ssm_box)
    ax.text(cx, 5.15, "RSSM Block", ha='center', va='center', fontsize=11.5, fontweight='bold', color='#000000', zorder=4)

    # CAFE
    create_rounded_box(ax, cx, 4.40, 2.9, 0.55, '#ECFDF5', '#059669', "CAFE", fontsize=10.2, lw=1.2, text_color='#065F46')

    # Rollout Planner
    create_rounded_box(ax, cx, 3.45, 2.9, 0.75, '#F5F3FF', '#7C3AED', "Rollout Planner", fontsize=10.2, lw=1.2, text_color='#5B21B6')

    draw_straight_arrow(ax, cx, 1.85, cx, 2.85)

    # Skip 1 around SSM
    draw_classic_skip(ax, cx, 2.35, cx + 1.65, 6.18, cx + 2.05, lw=1.6)

    # Direct Trunk Arrow
    draw_straight_arrow(ax, cx, 1.30, cx, 1.85)

    # Input Embedding
    create_rounded_box(ax, cx, 1.05, 3.4, 0.52, c_bg_embed, c_bd_embed, "Embedding", fontsize=11)
    draw_straight_arrow(ax, cx, 0.50, cx, 0.79)

    # Input Box
    create_rounded_box(ax, cx, 0.28, 3.4, 0.45, '#F8FAFC', '#0F172A', "Input", fontsize=11)

    # Autoregressive Loop
    draw_autoregressive_loop(ax, cx - 1.70, 12.35, cx - 1.70, 0.28, route_x=1.00, color='#DC2626', lw=1.8, label="Autoregressive Loop (t -> t+1)")

    out_dir = "/home/eun/neuroworld_lm/figures"
    plate_path = os.path.join(out_dir, "hokie_architecture_classic_plate.png")
    plate_pdf = os.path.join(out_dir, "hokie_architecture_classic_plate.pdf")
    plate_svg = os.path.join(out_dir, "hokie_architecture_classic_plate.svg")
    plt.savefig(plate_path, dpi=300, bbox_inches='tight')
    plt.savefig(plate_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(plate_svg, format='svg', bbox_inches='tight')
    plt.close()

    paper_fig_dir = "/home/eun/neuroworld_lm/paper/figures"
    if os.path.exists(paper_fig_dir):
        shutil.copy2(plate_path, os.path.join(paper_fig_dir, "hokie_architecture_classic_plate.png"))
        shutil.copy2(plate_pdf, os.path.join(paper_fig_dir, "hokie_architecture_classic_plate.pdf"))

    print(f"[✓] Single-Column Plate saved to: {plate_path}")


if __name__ == "__main__":
    render_true_architecture()
    render_classic_single_plate_true()
