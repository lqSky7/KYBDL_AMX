#!/usr/bin/env python3
"""
Generate all data-driven and architectural figures for the PQC-AMX paper.
Outputs SVG, PDF, and high-res PNG formats.
Style: Rounded corners, pastel pink/blue/purple palette, NO Reference C.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Arrow, FancyArrowPatch, Circle
import numpy as np
import os
import shutil

# ── Pastel palette ──────────────────────────────────────────────
PINK    = '#F2A7C3'   # AMX primary
BLUE    = '#A7C7E7'   # NEON primary
PURPLE  = '#C3A7E7'   # Cross-scheme / Hybrid
LPINK   = '#F9D1E0'
LBLUE   = '#D4E6F6'
LPURPLE = '#E1D1F6'
DPINK   = '#D4749E'
DBLUE   = '#6B9FCE'
DPURPLE = '#9B72CE'
GREY    = '#E8E8E8'
DGREY   = '#666666'
WHITE   = '#FFFFFF'
DARK    = '#2C3E50'

OUT_PROJECT = os.path.dirname(os.path.abspath(__file__))
OUT_DESKTOP = '/Users/ca5/Desktop/PQC_AMX_Diagrams'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.facecolor': WHITE,
    'figure.facecolor': WHITE,
    'savefig.facecolor': WHITE,
    'axes.edgecolor': '#CCCCCC',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': '#CCCCCC',
})


def rounded_bar(ax, x, height, width=0.35, color=BLUE, label=None, radius=0.06):
    """Draw a single bar with rounded top corners using FancyBboxPatch."""
    bar = FancyBboxPatch(
        (x - width/2, 0), width, height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=color, edgecolor='none', linewidth=0,
        label=label
    )
    ax.add_patch(bar)
    return bar


def rounded_bars(ax, positions, heights, width=0.35, color=BLUE, label=None, radius=0.06):
    """Draw multiple bars with rounded top corners."""
    patches = []
    for i, (x, h) in enumerate(zip(positions, heights)):
        p = rounded_bar(ax, x, h, width, color, label=label if i == 0 else None, radius=radius)
        patches.append(p)
    return patches


def save_figure(fig, base_name):
    """Save figure in both project directory and desktop folder in SVG, PDF, and PNG."""
    for out_dir in [OUT_PROJECT, OUT_DESKTOP]:
        try:
            os.makedirs(out_dir, exist_ok=True)
            fig.savefig(os.path.join(out_dir, f'{base_name}.svg'), format='svg', bbox_inches='tight')
            fig.savefig(os.path.join(out_dir, f'{base_name}.pdf'), format='pdf', bbox_inches='tight')
            fig.savefig(os.path.join(out_dir, f'{base_name}.png'), format='png', dpi=300, bbox_inches='tight')
        except OSError:
            pass
    plt.close(fig)
    print(f'  ✓ {base_name} (svg, pdf, png)')


# ═══════════════════════════════════════════════════════════════
# FIGURE 1: AMX Register Layout & mac16 Outer Product
# ═══════════════════════════════════════════════════════════════
from build_publication_figures import generate_amx_register_layout_svg, generate_methodology_blocks_svg, render_formats

def fig_amx_registers():
    svg = generate_amx_register_layout_svg()
    render_formats(svg, 'amx-register-layout')


# ═══════════════════════════════════════════════════════════════
# FIGURE 2: NTT Butterfly vs TMVP Schoolbook Outer-Product
# ═══════════════════════════════════════════════════════════════
def fig_ntt_butterfly():
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Title
    ax.text(5.0, 5.7, 'Algorithmic Contrast: NTT Butterfly vs. AMX Matrix TMVP',
            ha='center', va='center', fontsize=13, fontweight='bold', color=DARK)

    # Left Panel: NTT Butterfly
    panel_left = FancyBboxPatch((0.5, 0.6), 4.2, 4.7, boxstyle="round,pad=0.1,rounding_size=0.15",
                                facecolor=LBLUE, edgecolor=DBLUE, linewidth=1.5)
    ax.add_patch(panel_left)
    ax.text(2.6, 5.0, 'ML-KEM: O(n log n) NTT', ha='center', va='center',
            fontsize=11.5, fontweight='bold', color=DBLUE)
    ax.text(2.6, 4.65, 'Ring: Z_3329[x] / (x^256 + 1)', ha='center', va='center', fontsize=9, color=DARK)

    # Butterfly diagram sketch
    y_in = [4.1, 3.4, 2.7, 2.0]
    labels_in = ['a_0', 'a_1', 'a_2', 'a_3']
    for idx, (y_val, lbl) in enumerate(zip(y_in, labels_in)):
        ax.text(0.9, y_val, lbl, ha='center', va='center', fontsize=9, fontweight='bold', color=DARK)
        ax.plot([1.2, 2.2], [y_val, y_val], color=DBLUE, lw=1.5)

    # Cross connections (butterflies)
    ax.plot([2.2, 3.2], [4.1, 3.4], color=DBLUE, lw=1.5)
    ax.plot([2.2, 3.2], [3.4, 4.1], color=DBLUE, lw=1.5)
    ax.plot([2.2, 3.2], [2.7, 2.0], color=DBLUE, lw=1.5)
    ax.plot([2.2, 3.2], [2.0, 2.7], color=DBLUE, lw=1.5)

    for y_val in [4.1, 3.4, 2.7, 2.0]:
        ax.plot([3.2, 4.2], [y_val, y_val], color=DBLUE, lw=1.5)

    ax.text(2.6, 1.2, '• Pre-transformed keys (domain amortization)\n• Pointwise multiply: 256 mul + Inv-NTT\n• Heavily optimized in ARM NEON SIMD',
            ha='center', va='center', fontsize=8.5, color=DARK,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=WHITE, edgecolor='#B0C4DE', lw=1))

    # Right Panel: TMVP Matrix Outer Product
    panel_right = FancyBboxPatch((5.3, 0.6), 4.2, 4.7, boxstyle="round,pad=0.1,rounding_size=0.15",
                                 facecolor=LPINK, edgecolor=DPINK, linewidth=1.5)
    ax.add_patch(panel_right)
    ax.text(7.4, 5.0, 'AMX: O(n²) TMVP Outer Product', ha='center', va='center',
            fontsize=11.5, fontweight='bold', color=DPINK)
    ax.text(7.4, 4.65, 'Block-Toeplitz Convolution', ha='center', va='center', fontsize=9, color=DARK)

    # Grid sketch
    matrix_box = FancyBboxPatch((5.8, 2.1), 3.2, 2.1, boxstyle="round,pad=0.05,rounding_size=0.08",
                                facecolor=WHITE, edgecolor='#F5C6D6', linewidth=1.2)
    ax.add_patch(matrix_box)
    ax.text(7.4, 3.35, '32 × 32 Outer-Product Block', ha='center', va='center',
            fontsize=9.5, fontweight='bold', color=DPINK)
    ax.text(7.4, 2.8, 'Sum along anti-diagonals:\nc_k = Σ a_i · b_(k - i)', ha='center', va='center',
            fontsize=8.5, color=DARK)
    ax.text(7.4, 2.3, '(mac16 + EXTRH / VECINT flatten)', ha='center', va='center',
            fontsize=7.5, color=DGREY, fontstyle='italic')

    ax.text(7.4, 1.2, '• Operates directly on coefficients\n• Unamortized: 64 outer products for n=256\n• Cannot beat O(n log n) NTT on NEON',
            ha='center', va='center', fontsize=8.5, color=DARK,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=WHITE, edgecolor='#F5C6D6', lw=1))

    save_figure(fig, 'fft-butterfly')


# ═══════════════════════════════════════════════════════════════
# FIGURE 3: Methodology Pipeline Flowchart
# ═══════════════════════════════════════════════════════════════
def fig_methodology():
    svg = generate_methodology_blocks_svg()
    render_formats(svg, 'methodology-blocks')


# ═══════════════════════════════════════════════════════════════
# FIGURE 4: ML-KEM — NEON vs AMX cycle counts (log scale)
# ═══════════════════════════════════════════════════════════════
def fig_mlkem():
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    params  = ['ML-KEM-512', 'ML-KEM-768', 'ML-KEM-1024']
    ops     = ['KeyGen', 'Encaps', 'Decaps']

    neon = {
        'ML-KEM-512':  [13479, 14066, 20054],
        'ML-KEM-768':  [20933, 22087, 30939],
        'ML-KEM-1024': [29905, 31707, 45195],
    }
    amx = {
        'ML-KEM-512':  [413278, 622325, 832558],
        'ML-KEM-768':  [918005, 1234354, 1547285],
        'ML-KEM-1024': [1626327, 2049473, 2493322],
    }

    n_ops = len(ops)
    n_params = len(params)

    bar_w = 0.11
    pair_gap = 0.02
    pair_spacing = 0.32
    group_spacing = 1.2

    x_group_centers = np.arange(n_params) * group_spacing

    for i, p in enumerate(params):
        for j, op in enumerate(ops):
            pair_center = x_group_centers[i] + (j - 1) * pair_spacing
            xn = pair_center - bar_w / 2 - pair_gap / 2
            xa = pair_center + bar_w / 2 + pair_gap / 2
            rounded_bar(ax, xn, neon[p][j], bar_w, color=BLUE,
                        label='NEON NTT Baseline' if (i == 0 and j == 0) else None)
            rounded_bar(ax, xa, amx[p][j], bar_w, color=PINK,
                        label='AMX TMVP' if (i == 0 and j == 0) else None)

            ax.text(pair_center, 1800, op, ha='center', va='bottom',
                    fontsize=7.5, color=DGREY, fontstyle='italic')

            # Grayscaled approximate values on top of bars
            vn = neon[p][j]
            str_n = f"{vn/1e3:.1f}k"
            va = amx[p][j]
            str_a = f"{va/1e6:.2f}M" if va >= 1e6 else f"{va/1e3:.0f}k"

            ax.text(xn, vn * 1.15, str_n, ha='center', va='bottom',
                    fontsize=6.5, color=DGREY, fontweight='normal')
            ax.text(xa, va * 1.15, str_a, ha='center', va='bottom',
                    fontsize=6.5, color=DGREY, fontweight='normal')

    ax.set_yscale('log')
    ax.set_xticks(x_group_centers)
    ax.set_xticklabels(params)
    ax.set_ylabel('Cycle count (log scale)')
    ax.set_title('ML-KEM: NEON NTT vs AMX TMVP Cycle Counts on Apple M3')
    ax.set_ylim(1e3, 6.5e6)
    ax.set_xlim(
        x_group_centers[0] - pair_spacing - bar_w * 2,
        x_group_centers[-1] + pair_spacing + bar_w * 2
    )
    ax.legend(loc='upper left', framealpha=0.9)

    fig.tight_layout()
    save_figure(fig, 'fig-mlkem-cycles')


# ═══════════════════════════════════════════════════════════════
# FIGURE 5: MAYO — NEON vs AMX latency & speedup
# ═══════════════════════════════════════════════════════════════
def fig_mayo():
    fig, ax = plt.subplots(figsize=(7.8, 4.4))

    labels   = ['MAYO-1\n(64×66)', 'MAYO-2\n(64×78)', 'MAYO-3\n(96×99)', 'MAYO-5\n(128×133)']
    neon_lat = [129.67, 181.42, 435.10, 1054.80]
    amx_lat  = [87.94,  118.58, 272.23, 624.14]
    speedups = [1.47,   1.53,   1.60,   1.69]

    x = np.arange(len(labels))
    bar_w = 0.28
    pair_gap = 0.03

    rounded_bars(ax, x - bar_w/2 - pair_gap/2, neon_lat, bar_w, color=BLUE, label='NEON Baseline')
    rounded_bars(ax, x + bar_w/2 + pair_gap/2, amx_lat,  bar_w, color=PINK, label='AMX Accelerated')

    for i in range(len(labels)):
        ax.text(x[i] - bar_w/2 - pair_gap/2, neon_lat[i] * 1.08, f'{neon_lat[i]:.1f} µs',
                ha='center', va='bottom', fontsize=8, color=DBLUE, fontweight='bold')
        ax.text(x[i] + bar_w/2 + pair_gap/2, amx_lat[i] * 1.08, f'{amx_lat[i]:.1f} µs',
                ha='center', va='bottom', fontsize=8, color=DPINK, fontweight='bold')

    for i in range(len(labels)):
        s = speedups[i]
        badge_text = f'S = {s:.2f}×'
        ax.text(x[i], neon_lat[i] * 1.45, badge_text, ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#1b7837',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#d9f0d3', edgecolor='#1b7837', alpha=0.9, lw=1.0))

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Quadratic Form Evaluation Latency (µs, log scale)')
    ax.set_title('MAYO: NEON vs AMX Quadratic Form Latency on Apple M3')
    ax.set_ylim(40, 3000)
    ax.set_xlim(x[0] - 0.55, x[-1] + 0.55)
    ax.legend(loc='upper left', framealpha=0.9)

    fig.tight_layout()
    save_figure(fig, 'fig-mayo-speedup')


# ═══════════════════════════════════════════════════════════════
# FIGURE 6: Cross-scheme AMX/NEON speedup ratio (log scale)
# ═══════════════════════════════════════════════════════════════
def fig_crossscheme():
    fig, ax = plt.subplots(figsize=(8.2, 5.0))

    labels = [
        'MAYO-5',
        'MAYO-3',
        'MAYO-2',
        'MAYO-1',
        'sntrup761\n(ternary)',
        'sntrup761\n(dense)',
        'ML-KEM-512',
        'ML-KEM-768',
        'ML-KEM-1024',
        'HQC Sparse',
        'HQC Dense',
    ]
    speedups = [1.69, 1.60, 1.53, 1.47, 1.52, 0.92, 0.013, 0.013, 0.012, 0.0061, 0.00064]

    y = np.arange(len(labels))
    colors = []
    for s in speedups:
        if s >= 1.0:
            colors.append(PURPLE)
        elif s >= 0.5:
            colors.append(BLUE)
        else:
            colors.append(PINK)

    bar_h = 0.55

    for i, (yi, s) in enumerate(zip(y, speedups)):
        bar = FancyBboxPatch(
            (0.0003, yi - bar_h/2),
            s - 0.0003,
            bar_h,
            boxstyle="round,pad=0,rounding_size=0.1",
            facecolor=colors[i], edgecolor='none', alpha=0.85
        )
        ax.add_patch(bar)
        label_x = s * 1.35 if s < 0.1 else s * 1.18
        ax.text(label_x, yi, f'{s:.4f}×' if s < 0.01 else f'{s:.3f}×' if s < 0.1 else f'{s:.2f}×',
                va='center', ha='left', fontsize=9, fontweight='bold')

    ax.axvline(x=1.0, color=DPINK, linestyle='--', linewidth=2, alpha=0.8, zorder=5)
    ax.text(1.0, -0.45, 'Break-even (S = 1.0×)', ha='center', va='bottom',
            color=DPINK, fontsize=9, fontweight='bold')

    ax.set_xscale('log')
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Speedup S = t_NEON / t_AMX (log scale)')
    ax.set_title('Cross-Scheme AMX Feasibility: Speedup Over NEON on Apple M3')
    ax.set_xlim(0.0003, 5)
    ax.set_ylim(len(labels) - 0.35, -0.65)

    ax.axvspan(1.0, 5, alpha=0.06, color='green')
    ax.axvspan(0.0003, 1.0, alpha=0.06, color='red')
    ax.text(2.1, len(labels) - 0.55, 'AMX faster >>', ha='center', va='center', fontsize=8.5, fontweight='bold', color='#2e7d32')
    ax.text(0.01, len(labels) - 0.55, '<< NEON faster', ha='center', va='center', fontsize=8.5, fontweight='bold', color='#c62828')

    fig.tight_layout()
    save_figure(fig, 'fig-crossscheme-speedup')


# ═══════════════════════════════════════════════════════════════
# FIGURE 7: MAYO speedup vs matrix dimension (line plot)
# ═══════════════════════════════════════════════════════════════
def fig_mayo_trend():
    fig, ax = plt.subplots(figsize=(6.2, 3.8))

    labels = ['MAYO-1\n(64×66)', 'MAYO-2\n(64×78)', 'MAYO-3\n(96×99)', 'MAYO-5\n(128×133)']
    speedups = [1.47, 1.53, 1.60, 1.69]
    x = np.arange(len(labels))

    # Glow line + clean marker trend
    ax.plot(x, speedups, '-', color='#DDD6FE', linewidth=5, zorder=2, alpha=0.7)
    ax.plot(x, speedups, 'o-', color=DPURPLE, linewidth=2.4, markersize=8.5,
            markerfacecolor='#EDE9FE', markeredgecolor=DPURPLE, markeredgewidth=2.0, zorder=3)

    for xi, s in zip(x, speedups):
        ax.annotate(f'{s:.2f}×', (xi, s), textcoords='offset points',
                    xytext=(0, 11), ha='center', va='bottom',
                    fontsize=9.5, fontweight='bold', color=DPURPLE,
                    bbox=dict(boxstyle='round,pad=0.22,rounding_size=0.3',
                              facecolor='#F5F3FF', edgecolor='#DDD6FE', lw=1.2, zorder=4))

    ax.axhline(y=1.0, color='#EF4444', linestyle='--', linewidth=1.5, alpha=0.75,
               label='Break-even (S = 1.0×)', zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='medium', color=DARK)
    ax.set_xlabel('MAYO Parameter Set (Matrix Dimension m × n)', fontweight='bold', color=DARK, labelpad=8)
    ax.set_ylabel('Speedup S = t_NEON / t_AMX', fontweight='bold', color=DARK, labelpad=8)
    ax.set_title('MAYO: Speedup Scales with Matrix Size', fontweight='bold', fontsize=11.5, color=DARK, pad=12)
    ax.set_ylim(0.9, 1.88)
    ax.set_xlim(-0.45, len(labels) - 0.55)
    ax.legend(loc='lower right', framealpha=0.95, facecolor=WHITE, edgecolor='#E2E8F0', fontsize=9.5)

    fig.tight_layout()
    save_figure(fig, 'fig-mayo-trend')


# ═══════════════════════════════════════════════════════════════
# FIGURE 8: NTRU Prime (sntrup761) — NEON vs AMX (NO Ref C)
# ═══════════════════════════════════════════════════════════════
def fig_ntru():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    ops      = ['Ternary × Dense\n(KEM hot path)', 'Dense × Dense']
    neon_us  = [55.8, 78.1]
    amx_us   = [36.7, 84.6]
    speedups = [1.52, 0.92]

    x = np.arange(len(ops))
    bar_w = 0.28
    pair_gap = 0.03

    # Paired bars: NEON vs AMX only (NO Ref C)
    rounded_bars(ax, x - bar_w/2 - pair_gap/2, neon_us, bar_w, color=BLUE, label='NEON Baseline')
    rounded_bars(ax, x + bar_w/2 + pair_gap/2, amx_us,  bar_w, color=PINK, label='AMX Accelerated')

    for i in range(len(ops)):
        ax.text(x[i] - bar_w/2 - pair_gap/2, neon_us[i] + 1.5, f'{neon_us[i]:.1f} µs',
                ha='center', va='bottom', fontsize=9, color=DBLUE, fontweight='bold')
        ax.text(x[i] + bar_w/2 + pair_gap/2, amx_us[i] + 1.5, f'{amx_us[i]:.1f} µs',
                ha='center', va='bottom', fontsize=9, color=DPINK, fontweight='bold')

    for i in range(len(ops)):
        s = speedups[i]
        max_h = max(neon_us[i], amx_us[i])
        if s > 1.0:
            badge_text = f'S = {s:.2f}× (34% faster)'
            badge_color = '#1b7837'
            bg_color = '#d9f0d3'
        else:
            badge_text = f'S = {s:.2f}× (8% slower)'
            badge_color = '#762a83'
            bg_color = '#f7f7f7'

        ax.text(x[i], max_h + 10, badge_text, ha='center', va='bottom',
                fontsize=9.5, fontweight='bold', color=badge_color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=bg_color, edgecolor=badge_color, alpha=0.9, lw=1.2))

    ax.set_xticks(x)
    ax.set_xticklabels(ops)
    ax.set_ylabel('Kernel Latency (µs)')
    ax.set_title('sntrup761: NEON vs AMX Polynomial Multiplication Latency on Apple M3')
    ax.set_ylim(0, 115)
    ax.set_xlim(x[0] - 0.55, x[-1] + 0.55)
    ax.legend(loc='upper left', framealpha=0.9)

    fig.tight_layout()
    save_figure(fig, 'fig-ntru-latency')


# ═══════════════════════════════════════════════════════════════
# FIGURE 9: HQC-128 — NEON vs AMX (NO Ref C)
# ═══════════════════════════════════════════════════════════════
def fig_hqc():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    ops      = ['Dense × Dense', 'Sparse × Dense']
    neon_us  = [47, 11]
    amx_us   = [73430, 1812]
    speedups = [0.00064, 0.0061]

    x = np.arange(len(ops))
    bar_w = 0.28
    pair_gap = 0.03

    # Paired bars: NEON vs AMX only (NO Ref C)
    rounded_bars(ax, x - bar_w/2 - pair_gap/2, neon_us, bar_w, color=BLUE, label='NEON PMULL')
    rounded_bars(ax, x + bar_w/2 + pair_gap/2, amx_us,  bar_w, color=PINK, label='AMX')

    for i in range(len(ops)):
        ax.text(x[i] - bar_w/2 - pair_gap/2, neon_us[i] * 1.3, f'{neon_us[i]} µs',
                ha='center', va='bottom', fontsize=8.5, color=DBLUE, fontweight='bold')
        ax.text(x[i] + bar_w/2 + pair_gap/2, amx_us[i] * 1.3, f'{amx_us[i]:,} µs',
                ha='center', va='bottom', fontsize=8.5, color=DPINK, fontweight='bold')

    for i in range(len(ops)):
        s = speedups[i]
        slowdown = 1.0 / s
        ax.text(x[i] + bar_w/2 + pair_gap/2, amx_us[i] * 2.8, f'{slowdown:,.0f}× slower\nthan NEON',
                ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#CD5C5C')

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(ops)
    ax.set_ylabel('Kernel latency, µs (log scale)')
    ax.set_title('HQC-128: NEON PMULL vs AMX Kernel Latency on Apple M3')
    ax.set_ylim(2, 2.5e6)
    ax.set_xlim(x[0] - 0.55, x[-1] + 0.55)
    ax.legend(loc='upper right', framealpha=0.9)

    fig.tight_layout()
    save_figure(fig, 'fig-hqc-latency')


# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating paper figures (All Ref C removed)...')
    try:
        os.makedirs(OUT_DESKTOP, exist_ok=True)
    except OSError:
        pass
    fig_amx_registers()
    fig_ntt_butterfly()
    fig_methodology()
    fig_mlkem()
    fig_mayo()
    fig_crossscheme()
    fig_mayo_trend()
    fig_ntru()
    fig_hqc()
    print(f'\nAll diagrams successfully generated and copied to:\n  {OUT_DESKTOP}')
