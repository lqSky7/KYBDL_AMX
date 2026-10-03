#!/usr/bin/env python3
"""
Alternative (Non-Bar) Energy & Performance-per-Watt Visualizations for PQC-AMX.

Provides 3 advanced visual paradigms:
  Figure 1: Energy-Delay Pareto Vector Field (Log-Log Plane with Displacement Vectors)
            Shows simultaneous latency reduction and energy reduction pointing towards
            the optimal origin (bottom-left = Pareto dominant).
  Figure 2: Power-Time Cumulative Energy Area (Race-to-Sleep Dynamics)
            Visualizes instantaneous dynamic power P(t) over execution time; area under
            the curve directly depicts the 27.7% energy saved by AMX.
  Figure 3: Combined Dual-Panel Innovation Figure
            Panel A: Energy-Delay Pareto Vector Field.
            Panel B: Energy Savings (%) vs. Speedup (S) Quadrant Map.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.ticker as ticker

# ── Color Palette (Matching Paper's Clean Modern Aesthetic) ───────────────
PINK    = '#E57373'   # AMX point / fill
BLUE    = '#64B5F6'   # NEON point / fill
PURPLE  = '#BA68C8'   # Vector / Highlight
DARK    = '#2C3E50'   # Typography
DGREY   = '#546E7A'
LGREY   = '#ECEFF1'
WHITE   = '#FFFFFF'
GREEN   = '#2E7D32'
BG_GREEN= '#E8F5E9'
RED     = '#C62828'
BG_RED  = '#FFEBEE'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.facecolor': WHITE,
    'figure.facecolor': WHITE,
    'savefig.facecolor': WHITE,
    'axes.edgecolor': '#CFD8DC',
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.color': '#CFD8DC',
})

DATA = [
    # MAYO
    {"name": "MAYO-1", "cat": "win", "t_neon": 129.67, "t_amx": 87.94, "p_neon": 1.94, "p_amx": 2.36, "sub": "GF(16) 64×66"},
    {"name": "MAYO-2", "cat": "win", "t_neon": 181.42, "t_amx": 118.58, "p_neon": 1.95, "p_amx": 2.38, "sub": "GF(16) 64×78"},
    {"name": "MAYO-3", "cat": "win", "t_neon": 435.10, "t_amx": 272.23, "p_neon": 1.97, "p_amx": 2.40, "sub": "GF(16) 96×99"},
    {"name": "MAYO-5", "cat": "win", "t_neon": 1054.80, "t_amx": 624.14, "p_neon": 1.98, "p_amx": 2.42, "sub": "GF(16) 128×133"},

    # sntrup761
    {"name": "sntrup761 (Ternary)", "cat": "win", "t_neon": 55.80, "t_amx": 36.70, "p_neon": 1.92, "p_amx": 2.32, "sub": "KEM Hot Path"},
    {"name": "sntrup761 (Dense)", "cat": "loss", "t_neon": 78.10, "t_amx": 84.60, "p_neon": 1.93, "p_amx": 2.34, "sub": "Dense×Dense"},

    # ML-KEM
    {"name": "ML-KEM-512", "cat": "mismatch", "t_neon": 1.33, "t_amx": 99.80, "p_neon": 1.91, "p_amx": 2.35, "sub": "NTT vs TMVP"},
    {"name": "ML-KEM-768", "cat": "mismatch", "t_neon": 2.97, "t_amx": 224.90, "p_neon": 1.92, "p_amx": 2.36, "sub": "NTT vs TMVP"},
    {"name": "ML-KEM-1024", "cat": "mismatch", "t_neon": 4.64, "t_amx": 400.20, "p_neon": 1.93, "p_amx": 2.38, "sub": "NTT vs TMVP"},

    # HQC
    {"name": "HQC Sparse", "cat": "mismatch", "t_neon": 11.00, "t_amx": 1812.00, "p_neon": 1.91, "p_amx": 2.30, "sub": "Sparse×Dense"},
    {"name": "HQC Dense", "cat": "mismatch", "t_neon": 47.00, "t_amx": 73430.00, "p_neon": 1.92, "p_amx": 2.30, "sub": "Dense×Dense"},
]

for d in DATA:
    d["e_neon"] = d["t_neon"] * d["p_neon"]
    d["e_amx"]  = d["t_amx"]  * d["p_amx"]
    d["s"]      = d["t_neon"] / d["t_amx"]
    d["e_red"]  = (d["e_neon"] - d["e_amx"]) / d["e_neon"] * 100.0
    d["edp_gain"] = (d["e_neon"] * d["t_neon"]) / (d["e_amx"] * d["t_amx"])


# ==============================================================================
# VISUALIZATION 1: Dual-Panel (Pareto Vector Space + Energy-Speedup Quadrant)
# ==============================================================================
def generate_pareto_and_quadrant_figure():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 6.2), gridspec_kw={'width_ratios': [1.15, 1.0]})

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL A: Energy-Delay Pareto Space (Log-Log Plane with Displacement Vectors)
    # ──────────────────────────────────────────────────────────────────────────
    # Draw Iso-EDP diagonal contours (E * t = const)
    t_grid = np.logspace(0, 5, 200)
    for c_edp, label in [(1e2, "10² µJ·µs"), (1e4, "10⁴ µJ·µs"), (1e6, "10⁶ µJ·µs"), (1e8, "10⁸ µJ·µs"), (1e10, "10¹⁰ µJ·µs")]:
        e_grid = c_edp / t_grid
        valid = (e_grid >= 0.5) & (e_grid <= 5e5)
        if np.any(valid):
            ax1.plot(t_grid[valid], e_grid[valid], ':', color='#B0BEC5', lw=0.9, zorder=1)
            # Label near the top boundary
            idx = np.where(valid)[0][len(np.where(valid)[0]) // 3]
            ax1.text(t_grid[idx], e_grid[idx] * 1.15, f"Iso-EDP {label}", rotation=-36,
                     fontsize=7, color='#78909C', ha='center', va='bottom')

    # Desirable direction indicator
    ax1.annotate("Optimal Design Direction\n(Faster + Less Energy)",
                 xy=(3, 1.5), xytext=(25, 25),
                 arrowprops=dict(facecolor=GREEN, edgecolor=GREEN, width=1.8, headwidth=7, shrink=0.1),
                 fontsize=8.5, fontweight='bold', color=GREEN,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor=BG_GREEN, edgecolor=GREEN, lw=1.0),
                 ha='center', va='center')

    # Plot NEON and AMX points with connecting displacement vectors
    for d in DATA:
        tn, en = d["t_neon"], d["e_neon"]
        ta, ea = d["t_amx"], d["e_amx"]

        is_winner = d["cat"] == "win"
        vec_color = GREEN if is_winner else RED
        vec_width = 1.6 if is_winner else 1.2
        head_w = 6 if is_winner else 4.5

        # Plot NEON baseline (Blue Circle)
        ax1.scatter(tn, en, color=BLUE, s=48, edgecolors='#1E88E5', lw=1.2, zorder=5)

        # Plot AMX accelerated (Pink Square)
        ax1.scatter(ta, ea, color=PINK, s=55, marker='s', edgecolors='#C2185B', lw=1.2, zorder=5)

        # Draw vector from NEON to AMX
        ax1.annotate('', xy=(ta, ea), xytext=(tn, en),
                     arrowprops=dict(arrowstyle="-|>", color=vec_color, lw=vec_width,
                                     mutation_scale=11, shrinkA=3, shrinkB=3),
                     zorder=4)

        # Labeling
        if is_winner:
            offset_y = 1.28 if "MAYO-5" in d["name"] or "MAYO-1" in d["name"] else 0.72
            ax1.text(ta * 0.95, ea * offset_y, f"{d['name']}\n(-{d['e_red']:.1f}% E, {d['s']:.2f}×)",
                     fontsize=8.0, fontweight='bold', color=DARK, ha='right' if offset_y > 1 else 'center')
        elif "Dense" in d["name"] and "sntrup" in d["name"]:
            ax1.text(ta * 1.15, ea * 1.05, f"{d['name']}", fontsize=7.5, color=DGREY)
        elif "ML-KEM-512" in d["name"]:
            ax1.text(ta * 1.15, ea * 0.95, f"{d['name']}\n(+{abs(d['e_red']):.0f}% E)",
                     fontsize=7.5, color=RED, va='top')
        elif "HQC Dense" in d["name"]:
            ax1.text(ta * 0.9, ea * 1.25, f"{d['name']}", fontsize=7.5, color=RED, ha='right')

    # Legend for Panel A
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=BLUE, markeredgecolor='#1E88E5', markersize=8, label='NEON Baseline'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=PINK, markeredgecolor='#C2185B', markersize=8, label='AMX Accelerated'),
        plt.Line2D([0], [0], color=GREEN, lw=2, label='AMX Wins (Faster & Saves Energy)'),
        plt.Line2D([0], [0], color=RED, lw=1.5, linestyle='--', label='AMX Penalized (Slower & High Energy)'),
    ]
    ax1.legend(handles=legend_elements, loc='upper left', framealpha=0.95, facecolor=WHITE, edgecolor='#CFD8DC', fontsize=8.2)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(0.8, 1.2e5)
    ax1.set_ylim(0.8, 3e5)
    ax1.set_xlabel('Kernel Execution Latency (µs, log scale)', fontweight='bold', color=DARK, labelpad=6)
    ax1.set_ylabel('Energy per Operation (µJ, log scale)', fontweight='bold', color=DARK, labelpad=6)
    ax1.set_title('(A) Energy-Delay Pareto Space (Vector Shifts: NEON -> AMX)',
                  fontweight='bold', fontsize=11, color=DARK, pad=10)

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL B: Energy Savings (%) vs. Speedup (S) Quadrant Map
    # ──────────────────────────────────────────────────────────────────────────
    # The 4 quadrants defined by Break-even Speedup (S = 1.0) and Energy Neutral (0%)
    ax2.axvline(1.0, color='#90A4AE', linestyle='--', lw=1.5, zorder=2)
    ax2.axhline(0.0, color='#90A4AE', linestyle='--', lw=1.5, zorder=2)

    # Quadrant Shading
    # Upper-Right: SWEET SPOT (Faster + Energy Efficient)
    ax2.axhspan(0, 35, xmin=0.38, xmax=1.0, color='#E8F5E9', alpha=0.6, zorder=1)
    ax2.text(1.35, 31, "AMX Sweet Spot\n(Faster + Less Energy)",
             fontsize=9.0, fontweight='bold', color=GREEN,
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#C8E6C9', edgecolor=GREEN, lw=1.0))

    # Lower-Left: LOSS ZONE (Slower + Energy Penalty)
    ax2.axhspan(-100, 0, xmin=0.0, xmax=0.38, color='#FFEBEE', alpha=0.5, zorder=1)
    ax2.text(0.0025, -78, "AMX Penalty Zone\n(Slower + Higher Energy)",
             fontsize=8.5, fontweight='bold', color=RED,
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#FFCDD2', edgecolor=RED, lw=1.0))

    # Plot bubble points for all operations
    for d in DATA:
        s = d["s"]
        e_red = d["e_red"]
        # Clamp very large negative energy reduction to -95% for display on map
        clamped_e = max(e_red, -92.0)

        is_winner = d["cat"] == "win"
        bubble_color = PURPLE if is_winner else '#EF5350'
        bubble_size = 180 if is_winner else 110

        ax2.scatter(s, clamped_e, color=bubble_color, s=bubble_size, edgecolors=WHITE, lw=1.5, zorder=5)

        # Label points
        if is_winner:
            ax2.annotate(f"{d['name']}\nS={s:.2f}×, E={e_red:.1f}%",
                         xy=(s, clamped_e), xytext=(s * 1.03, clamped_e - 3.8),
                         fontsize=8.0, fontweight='bold', color=DARK,
                         ha='left', va='top')
        elif "Dense" in d["name"] and "sntrup" in d["name"]:
            ax2.annotate(f"{d['name']}\nS={s:.2f}×, E={e_red:.1f}%",
                         xy=(s, clamped_e), xytext=(s * 1.05, clamped_e - 2.5),
                         fontsize=7.5, color=DARK)
        elif "ML-KEM-512" in d["name"]:
            ax2.annotate("ML-KEM (All)", xy=(s, clamped_e), xytext=(s * 1.5, clamped_e + 4.0),
                         fontsize=8.0, color=RED, fontweight='bold')
        elif "HQC" in d["name"] and "Dense" in d["name"]:
            ax2.annotate("HQC (All)", xy=(s, clamped_e), xytext=(s * 1.8, clamped_e + 4.0),
                         fontsize=8.0, color=RED, fontweight='bold')

    ax2.set_xscale('log')
    ax2.set_xlim(0.0004, 2.5)
    ax2.set_ylim(-98, 38)
    ax2.set_xlabel('Kernel Speedup S = t_NEON / t_AMX (log scale)', fontweight='bold', color=DARK, labelpad=6)
    ax2.set_ylabel('Energy Reduction (%)  [Positive = Energy Saved]', fontweight='bold', color=DARK, labelpad=6)
    ax2.set_title('(B) Energy-Speedup Feasibility Quadrant Map',
                  fontweight='bold', fontsize=11, color=DARK, pad=10)

    # Clean axes formatter
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:g}×'))
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{int(y)}%'))

    fig.tight_layout()

    # Save to both project root and Desktop
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(f"fig-energy-pareto-quadrant.{ext}", format=ext, dpi=300, bbox_inches='tight')
        desktop_dir = "/Users/ca5/Desktop/PQC_AMX_Diagrams"
        if os.path.isdir(desktop_dir):
            fig.savefig(os.path.join(desktop_dir, f"fig-energy-pareto-quadrant.{ext}"),
                        format=ext, dpi=300, bbox_inches='tight')

    plt.close(fig)
    print("  ✓ Generated: fig-energy-pareto-quadrant (pdf, svg, png)")


# ==============================================================================
# VISUALIZATION 2: Dynamic Power Profile & Race-to-Sleep Trajectory Area
# ==============================================================================
def generate_race_to_sleep_figure():
    """
    Shows the power-time area curve for MAYO-5:
    Demonstrates that AMX draws slightly higher power (2.42W vs 1.98W)
    but finishes so much earlier (624 us vs 1054 us) that the total area
    (Energy = integral P dt) is 27.7% smaller!
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.8), sharey=True)

    # ── SUBPLOT 1: MAYO-5 Race-to-Sleep ──────────────────────────────────────
    t_neon_m5 = 1054.8
    t_amx_m5  = 624.1
    p_neon_m5 = 1.98
    p_amx_m5  = 2.42
    p_idle    = 0.38

    # Time steps for plotting step curves
    t_axis_m5 = np.linspace(0, 1200, 1000)

    # Power curves
    p_curve_neon_m5 = np.where(t_axis_m5 <= t_neon_m5, p_neon_m5, 0.0)
    p_curve_amx_m5  = np.where(t_axis_m5 <= t_amx_m5, p_amx_m5, 0.0)

    # Fill areas (Energy)
    ax1.fill_between(t_axis_m5, 0, p_curve_neon_m5, color=BLUE, alpha=0.35, label='NEON Baseline Area (2,088 µJ)')
    ax1.plot(t_axis_m5, p_curve_neon_m5, color='#1E88E5', lw=2.2)

    ax1.fill_between(t_axis_m5, 0, p_curve_amx_m5, color=PINK, alpha=0.45, label='AMX Accelerated Area (1,510 µJ)')
    ax1.plot(t_axis_m5, p_curve_amx_m5, color='#D81B60', lw=2.2)

    # Shaded Saved Energy Region
    t_saved = np.linspace(t_amx_m5, t_neon_m5, 200)
    ax1.fill_between(t_saved, 0, p_neon_m5, color='#A5D6A7', alpha=0.8,
                     label='Net Energy Saved by AMX (578 µJ = 27.7%)')

    ax1.axvline(t_amx_m5, color='#D81B60', linestyle='--', lw=1.2)
    ax1.text(t_amx_m5 - 15, 2.52, f"AMX Done\n({t_amx_m5:.0f} µs)", ha='right', va='bottom',
             fontsize=8.5, fontweight='bold', color='#D81B60')

    ax1.axvline(t_neon_m5, color='#1E88E5', linestyle='--', lw=1.2)
    ax1.text(t_neon_m5 + 15, 2.1, f"NEON Done\n({t_neon_m5:.0f} µs)", ha='left', va='bottom',
             fontsize=8.5, fontweight='bold', color='#1E88E5')

    ax1.text(840, 1.0, "27.7%\nEnergy Saved\nvia Race-to-Sleep", ha='center', va='center',
             fontsize=9.0, fontweight='bold', color=GREEN)

    ax1.set_xlim(0, 1250)
    ax1.set_ylim(0, 2.85)
    ax1.set_xlabel('Execution Timeline (µs)', fontweight='bold', color=DARK, labelpad=6)
    ax1.set_ylabel('Isolated Dynamic Core Power (Watts)', fontweight='bold', color=DARK, labelpad=6)
    ax1.set_title('(A) MAYO-5: Dynamic Power × Time (Area = Energy)',
                  fontweight='bold', fontsize=11, color=DARK, pad=10)
    ax1.legend(loc='lower left', framealpha=0.95, facecolor=WHITE, edgecolor='#CFD8DC', fontsize=8.2)

    # ── SUBPLOT 2: sntrup761 Ternary Race-to-Sleep ───────────────────────────
    t_neon_ntru = 55.8
    t_amx_ntru  = 36.7
    p_neon_ntru = 1.92
    p_amx_ntru  = 2.32

    t_axis_ntru = np.linspace(0, 75, 1000)
    p_curve_neon_ntru = np.where(t_axis_ntru <= t_neon_ntru, p_neon_ntru, 0.0)
    p_curve_amx_ntru  = np.where(t_axis_ntru <= t_amx_ntru, p_amx_ntru, 0.0)

    ax2.fill_between(t_axis_ntru, 0, p_curve_neon_ntru, color=BLUE, alpha=0.35, label='NEON Baseline Area (107 µJ)')
    ax2.plot(t_axis_ntru, p_curve_neon_ntru, color='#1E88E5', lw=2.2)

    ax2.fill_between(t_axis_ntru, 0, p_curve_amx_ntru, color=PINK, alpha=0.45, label='AMX Accelerated Area (85 µJ)')
    ax2.plot(t_axis_ntru, p_curve_amx_ntru, color='#D81B60', lw=2.2)

    t_saved_ntru = np.linspace(t_amx_ntru, t_neon_ntru, 200)
    ax2.fill_between(t_saved_ntru, 0, p_neon_ntru, color='#A5D6A7', alpha=0.8,
                     label='Net Energy Saved (22 µJ = 20.5%)')

    ax2.axvline(t_amx_ntru, color='#D81B60', linestyle='--', lw=1.2)
    ax2.text(t_amx_ntru - 1.5, 2.42, f"AMX Done\n({t_amx_ntru:.1f} µs)", ha='right', va='bottom',
             fontsize=8.5, fontweight='bold', color='#D81B60')

    ax2.axvline(t_neon_ntru, color='#1E88E5', linestyle='--', lw=1.2)
    ax2.text(t_neon_ntru + 1.5, 2.05, f"NEON Done\n({t_neon_ntru:.1f} µs)", ha='left', va='bottom',
             fontsize=8.5, fontweight='bold', color='#1E88E5')

    ax2.text(46.2, 0.95, "20.5%\nSaved", ha='center', va='center',
             fontsize=8.8, fontweight='bold', color=GREEN)

    ax2.set_xlim(0, 72)
    ax2.set_xlabel('Execution Timeline (µs)', fontweight='bold', color=DARK, labelpad=6)
    ax2.set_title('(B) sntrup761 Ternary: Dynamic Power × Time (Area = Energy)',
                  fontweight='bold', fontsize=11, color=DARK, pad=10)
    ax2.legend(loc='lower left', framealpha=0.95, facecolor=WHITE, edgecolor='#CFD8DC', fontsize=8.2)

    fig.tight_layout()

    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(f"fig-energy-race-to-sleep.{ext}", format=ext, dpi=300, bbox_inches='tight')
        desktop_dir = "/Users/ca5/Desktop/PQC_AMX_Diagrams"
        if os.path.isdir(desktop_dir):
            fig.savefig(os.path.join(desktop_dir, f"fig-energy-race-to-sleep.{ext}"),
                        format=ext, dpi=300, bbox_inches='tight')

    plt.close(fig)
    print("  ✓ Generated: fig-energy-race-to-sleep (pdf, svg, png)")


def main():
    print("Generating alternative publication-quality visual formats...")
    generate_pareto_and_quadrant_figure()
    generate_race_to_sleep_figure()
    print("✅ All alternative figures successfully generated!")


if __name__ == '__main__':
    main()
