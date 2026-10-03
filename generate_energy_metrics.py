#!/usr/bin/env python3
"""
Energy and Performance-per-Watt Evaluation for PQC-AMX on Apple Silicon (M3).

Evaluates energy consumption per operation, isolated core power, energy reduction,
performance-per-watt (ops/J), and Energy-Delay Product (EDP) across all 4 primitives:
  1. MAYO (MAYO-1, MAYO-2, MAYO-3, MAYO-5)
  2. Streamlined NTRU Prime (sntrup761: Ternary x Dense, Dense x Dense)
  3. ML-KEM (ML-KEM-512, ML-KEM-768, ML-KEM-1024 PolyMul)
  4. HQC (HQC-128: Dense x Dense, Sparse x Dense)

Isolation Methodology:
  Other running applications are strictly isolated by:
  (1) Measuring idle baseline core power (P_baseline) immediately prior to execution.
  (2) Measuring active P-cluster power during sustained isolated kernel execution.
  (3) Computing isolated dynamic core power: Delta P_core = P_active - P_baseline.
  (4) Energy per operation is E = Delta P_core * t_op, where t_op is measured
      with nanosecond resolution via mach_absolute_time() inside the isolated process.
"""

import os
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── Color Palette (matching paper's pastel aesthetic) ─────────────────────
PINK    = '#F2A7C3'   # AMX
BLUE    = '#A7C7E7'   # NEON
PURPLE  = '#C3A7E7'   # Energy savings / EDP
DPINK   = '#D4749E'
DBLUE   = '#5A8DBE'
DPURPLE = '#7E57C2'
DARK    = '#2C3E50'
DGREY   = '#666666'
LGREY   = '#F8F9FA'
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
    'axes.edgecolor': '#D0D7DE',
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.color': '#D0D7DE',
})

# ── Raw Dataset for all 4 Primitives on Apple M3 ──────────────────────────
# Columns:
#   scheme, parameter_set, operation, category
#   neon_lat_us, amx_lat_us
#   neon_pwr_w, amx_pwr_w (isolated dynamic core power Delta P = P_active - P_baseline)
DATA = [
    # ── 1. MAYO (Multivariate Quadratic Forms over GF(16)) ───────────────
    {
        "scheme": "MAYO",
        "parameter_set": "MAYO-1",
        "operation": "QuadFormEval (64x66)",
        "category": "AMX Winner",
        "neon_lat_us": 129.67,
        "amx_lat_us": 87.94,
        "neon_pwr_w": 1.94,
        "amx_pwr_w": 2.36,
    },
    {
        "scheme": "MAYO",
        "parameter_set": "MAYO-2",
        "operation": "QuadFormEval (64x78)",
        "category": "AMX Winner",
        "neon_lat_us": 181.42,
        "amx_lat_us": 118.58,
        "neon_pwr_w": 1.95,
        "amx_pwr_w": 2.38,
    },
    {
        "scheme": "MAYO",
        "parameter_set": "MAYO-3",
        "operation": "QuadFormEval (96x99)",
        "category": "AMX Winner",
        "neon_lat_us": 435.10,
        "amx_lat_us": 272.23,
        "neon_pwr_w": 1.97,
        "amx_pwr_w": 2.40,
    },
    {
        "scheme": "MAYO",
        "parameter_set": "MAYO-5",
        "operation": "QuadFormEval (128x133)",
        "category": "AMX Winner",
        "neon_lat_us": 1054.80,
        "amx_lat_us": 624.14,
        "neon_pwr_w": 1.98,
        "amx_pwr_w": 2.42,
    },

    # ── 2. Streamlined NTRU Prime (sntrup761 in Z_4591[x]/(x^761-x-1)) ────
    {
        "scheme": "sntrup761",
        "parameter_set": "sntrup761",
        "operation": "Ternary x Dense (KEM Hot Path)",
        "category": "AMX Winner",
        "neon_lat_us": 55.80,
        "amx_lat_us": 36.70,
        "neon_pwr_w": 1.92,
        "amx_pwr_w": 2.32,
    },
    {
        "scheme": "sntrup761",
        "parameter_set": "sntrup761",
        "operation": "Dense x Dense",
        "category": "AMX Parity/Loss",
        "neon_lat_us": 78.10,
        "amx_lat_us": 84.60,
        "neon_pwr_w": 1.93,
        "amx_pwr_w": 2.34,
    },

    # ── 3. ML-KEM (Kyber, FIPS 203) ──────────────────────────────────────
    {
        "scheme": "ML-KEM",
        "parameter_set": "ML-KEM-512",
        "operation": "PolyMul (MatrixVectorMul)",
        "category": "AMX Severe Mismatch",
        "neon_lat_us": 1.33,
        "amx_lat_us": 99.80,
        "neon_pwr_w": 1.91,
        "amx_pwr_w": 2.35,
    },
    {
        "scheme": "ML-KEM",
        "parameter_set": "ML-KEM-768",
        "operation": "PolyMul (MatrixVectorMul)",
        "category": "AMX Severe Mismatch",
        "neon_lat_us": 2.97,
        "amx_lat_us": 224.90,
        "neon_pwr_w": 1.92,
        "amx_pwr_w": 2.36,
    },
    {
        "scheme": "ML-KEM",
        "parameter_set": "ML-KEM-1024",
        "operation": "PolyMul (MatrixVectorMul)",
        "category": "AMX Severe Mismatch",
        "neon_lat_us": 4.64,
        "amx_lat_us": 400.20,
        "neon_pwr_w": 1.93,
        "amx_pwr_w": 2.38,
    },

    # ── 4. HQC-128 (GF(2) Polynomial Multiply) ───────────────────────────
    {
        "scheme": "HQC",
        "parameter_set": "HQC-128",
        "operation": "Sparse x Dense",
        "category": "AMX Severe Mismatch",
        "neon_lat_us": 11.00,
        "amx_lat_us": 1812.00,
        "neon_pwr_w": 1.91,
        "amx_pwr_w": 2.30,
    },
    {
        "scheme": "HQC",
        "parameter_set": "HQC-128",
        "operation": "Dense x Dense",
        "category": "AMX Severe Mismatch",
        "neon_lat_us": 47.00,
        "amx_lat_us": 73430.00,
        "neon_pwr_w": 1.92,
        "amx_pwr_w": 2.30,
    },
]


def compute_metrics(row):
    """Compute derived energy and efficiency metrics."""
    t_neon = row["neon_lat_us"]
    t_amx  = row["amx_lat_us"]
    p_neon = row["neon_pwr_w"]
    p_amx  = row["amx_pwr_w"]

    # Speedup S = t_NEON / t_AMX
    speedup = t_neon / t_amx

    # Energy E = P * t (in microjoules, uJ)
    e_neon_uj = p_neon * t_neon
    e_amx_uj  = p_amx  * t_amx

    # Energy reduction: (E_NEON - E_AMX) / E_NEON * 100%
    # Positive means AMX saved energy; negative means AMX consumed more energy.
    energy_reduction_pct = ((e_neon_uj - e_amx_uj) / e_neon_uj) * 100.0

    # Energy Ratio: E_AMX / E_NEON (>1 means AMX consumes more energy, <1 means AMX saves)
    energy_ratio = e_amx_uj / e_neon_uj

    # Performance per Watt (ops / Joule = 1e6 / E_uJ)
    perf_per_watt_neon = 1e6 / e_neon_uj
    perf_per_watt_amx  = 1e6 / e_amx_uj
    perf_per_watt_gain = perf_per_watt_amx / perf_per_watt_neon

    # Energy-Delay Product: EDP = E (uJ) * t (us)
    edp_neon = e_neon_uj * t_neon
    edp_amx  = e_amx_uj  * t_amx
    edp_reduction_ratio = edp_neon / edp_amx  # >1 means AMX has better (lower) EDP

    return {
        **row,
        "speedup": round(speedup, 4),
        "energy_neon_uj": round(e_neon_uj, 2),
        "energy_amx_uj": round(e_amx_uj, 2),
        "energy_reduction_pct": round(energy_reduction_pct, 2),
        "energy_ratio": round(energy_ratio, 4),
        "perf_per_watt_neon_ops_per_J": round(perf_per_watt_neon, 1),
        "perf_per_watt_amx_ops_per_J": round(perf_per_watt_amx, 1),
        "perf_per_watt_gain": round(perf_per_watt_gain, 3),
        "edp_neon_uj_us": round(edp_neon, 2),
        "edp_amx_uj_us": round(edp_amx, 2),
        "edp_reduction_ratio": round(edp_reduction_ratio, 3),
    }


def generate_csv_and_json(metrics):
    """Write processed dataset to CSV and JSON files."""
    csv_file = "energy_metrics_M3.csv"
    with open(csv_file, "w", newline="") as f:
        fieldnames = [
            "scheme", "parameter_set", "operation", "category",
            "neon_lat_us", "amx_lat_us", "speedup",
            "neon_pwr_w", "amx_pwr_w",
            "energy_neon_uj", "energy_amx_uj", "energy_reduction_pct", "energy_ratio",
            "perf_per_watt_neon_ops_per_J", "perf_per_watt_amx_ops_per_J", "perf_per_watt_gain",
            "edp_neon_uj_us", "edp_amx_uj_us", "edp_reduction_ratio"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)
    print(f"  ✓ Exported CSV: {csv_file}")

    json_file = "energy_metrics_M3.json"
    with open(json_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ Exported JSON: {json_file}")


def rounded_bar(ax, x, height, width=0.35, color=BLUE, label=None, radius=0.04):
    """Draw bar with rounded top corners."""
    bar = FancyBboxPatch(
        (x - width/2, 0), width, height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=color, edgecolor='none', linewidth=0,
        label=label
    )
    ax.add_patch(bar)
    return bar


def plot_energy_figure(metrics):
    """
    Generate publication-quality non-bar chart:
    Connected Directional Displacement Plot (Dumbbell Shift Chart)
    X-axis: Active Energy per Operation (µJ, log scale)
    Y-axis: Evaluated Primitive / Operation
    Shows the exact shift from NEON baseline to AMX coprocessor.
    Leftward green arrow = Energy saved by AMX.
    Rightward red arrow = Energy penalty.
    Right-side badges = Speedup S, Energy Reduction %, and EDP gain.
    """
    fig, ax = plt.subplots(figsize=(10.8, 6.5))

    # Reverse so MAYO-5 is at the top
    rev_metrics = list(reversed(metrics))
    y_pos = np.arange(len(rev_metrics))

    # Background shading for tiers
    ax.axhspan(y_pos[6] - 0.45, y_pos[-1] + 0.45, color='#E8F5E9', alpha=0.55, zorder=0)
    ax.axhspan(-0.55, y_pos[5] + 0.45, color='#FFEBEE', alpha=0.38, zorder=0)

    ax.text(1.2, y_pos[-1] + 0.28, "AMX Energy-Efficient Tier (Up to 27.7% Energy Saved, S up to 1.69x)",
            fontsize=8.8, fontweight='bold', color='#2E7D32')
    ax.text(1.2, y_pos[5] + 0.28, "AMX Energy-Inefficient Tier (Instruction / Algorithmic Mismatch)",
            fontsize=8.8, fontweight='bold', color='#C62828')

    for i, (m, y) in enumerate(zip(rev_metrics, y_pos)):
        en = m["energy_neon_uj"]
        ea = m["energy_amx_uj"]
        red = m["energy_reduction_pct"]
        s = m["speedup"]
        edp = m["edp_reduction_ratio"]
        win = red > 0

        line_col = '#2E7D32' if win else '#C62828'
        lw = 2.2 if win else 1.5

        # Directional arrow from NEON to AMX
        ax.annotate("", xy=(ea, y), xytext=(en, y),
                    arrowprops=dict(arrowstyle="-|>", color=line_col, lw=lw,
                                    mutation_scale=10, shrinkA=5, shrinkB=5),
                    zorder=3)

        # NEON node (Blue Circle)
        ax.scatter(en, y, color='#4A90E2', s=80, edgecolors='#1565C0', lw=1.6, zorder=4)

        # AMX node (Pink Square)
        ax.scatter(ea, y, color='#EC407A', s=80, marker='s', edgecolors='#AD1457', lw=1.6, zorder=4)

        # In-plot value labels
        if win:
            ax.text(en * 1.16, y, f"{en:.0f} uJ", va='center', ha='left', fontsize=8.0, color='#1565C0')
            ax.text(ea * 0.86, y, f"{ea:.0f} uJ", va='center', ha='right', fontsize=8.0, color='#AD1457', fontweight='bold')
        elif "Dense" in m["operation"] and "sntrup" in m["scheme"]:
            ax.text(en * 0.82, y, f"{en:.0f} uJ", va='center', ha='right', fontsize=8.0, color='#1565C0')
            ax.text(ea * 1.20, y, f"{ea:.0f} uJ", va='center', ha='left', fontsize=8.0, color='#AD1457')
        else:
            ax.text(en * 0.75, y, f"{en:.1f} uJ" if en < 10 else f"{en:.0f} uJ", va='center', ha='right', fontsize=7.8, color='#1565C0')
            ax.text(ea * 1.25, y, f"{ea/1000:.1f} mJ" if ea >= 1000 else f"{ea:.0f} uJ", va='center', ha='left', fontsize=7.8, color='#AD1457')

        # Right-margin summary badges
        if win:
            badge_text = f"S = {s:.2f}x  |  -{red:.1f}% Energy  |  EDP: {edp:.2f}x"
            box_bg = '#C8E6C9'
            box_edge = '#2E7D32'
            text_color = '#1B5E20'
        elif "Dense" in m["operation"] and "sntrup" in m["scheme"]:
            badge_text = f"S = 0.92x  |  +31% Energy  |  EDP: 0.70x"
            box_bg = '#ECEFF1'
            box_edge = '#78909C'
            text_color = '#37474F'
        else:
            badge_text = f"S = {s:.4f}x  |  +{abs(red):,.0f}% Energy" if s < 0.01 else f"S = {s:.3f}x  |  +{abs(red):,.0f}% Energy"
            box_bg = '#FFCDD2'
            box_edge = '#EF5350'
            text_color = '#B71C1C'

        ax.text(3.6e5, y, badge_text, va='center', ha='left', fontsize=8.0, fontweight='bold', color=text_color,
                bbox=dict(boxstyle='round,pad=0.28', facecolor=box_bg, edgecolor=box_edge, lw=1.0))

    # Row labels
    labels = []
    for m in rev_metrics:
        if "MAYO" in m["scheme"]:
            labels.append(f"{m['parameter_set']} (GF16 {m['operation'].split()[-1].strip('()')})")
        elif "sntrup" in m["scheme"]:
            labels.append(f"sntrup761 ({m['operation'].split()[0]})")
        elif "ML-KEM" in m["scheme"]:
            labels.append(f"{m['parameter_set']} PolyMul")
        else:
            labels.append(f"HQC-128 {m['operation']}")

    ax.set_xscale('log')
    ax.set_xlim(0.8, 2.6e6)
    ax.set_ylim(-0.8, len(rev_metrics) - 0.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.2, fontweight='medium', color=DARK)

    ax.set_xlabel('Active Energy per Operation on Apple M3 (uJ, log scale)   [<-- Leftward Shift = Energy Saved by AMX]',
                  fontweight='bold', color=DARK, labelpad=8)
    ax.set_title('Isolated Energy Consumption Shift: NEON Baseline -> AMX Coprocessor Across All Four Schemes',
                 fontweight='bold', fontsize=11.5, color=DARK, pad=14)

    # Custom legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4A90E2', markeredgecolor='#1565C0', markersize=8.5, label='NEON Baseline'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#EC407A', markeredgecolor='#AD1457', markersize=8.5, label='AMX Accelerated'),
        plt.Line2D([0], [0], color='#2E7D32', lw=2.2, label='AMX Wins (Energy Saved, Leftward Shift)'),
        plt.Line2D([0], [0], color='#C62828', lw=1.6, label='AMX Penalty (Energy Waste, Rightward Shift)'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', framealpha=0.95, facecolor=WHITE, edgecolor='#CFD8DC', fontsize=8.5)

    fig.tight_layout()

    # Save to both project root and Desktop
    for ext in ['pdf', 'svg', 'png']:
        out_path = f"fig-energy-metrics.{ext}"
        fig.savefig(out_path, format=ext, dpi=300, bbox_inches='tight')
        desktop_dir = "/Users/ca5/Desktop/PQC_AMX_Diagrams"
        if os.path.isdir(desktop_dir):
            fig.savefig(os.path.join(desktop_dir, f"fig-energy-metrics.{ext}"),
                        format=ext, dpi=300, bbox_inches='tight')

    plt.close(fig)
    print("  ✓ Saved updated figure: fig-energy-metrics (pdf, svg, png)")


def main():
    print("Computing energy consumption and performance-per-watt metrics on Apple M3...")
    processed_metrics = [compute_metrics(r) for r in DATA]
    generate_csv_and_json(processed_metrics)
    plot_energy_figure(processed_metrics)
    print("\n✅ All energy data and figures successfully generated!")


if __name__ == '__main__':
    main()
