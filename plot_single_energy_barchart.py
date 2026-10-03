#!/usr/bin/env python3
"""
Single Bar Chart: Cross-Scheme Energy per Operation on Apple M3 (NEON vs. AMX)
Generates high-resolution publication-ready PDF, SVG, and PNG.
"""

import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.facecolor': '#FFFFFF',
    'figure.facecolor': '#FFFFFF',
    'savefig.facecolor': '#FFFFFF',
    'axes.edgecolor': '#CFD8DC',
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.color': '#CFD8DC',
})

ROWS = [
    {"name": "MAYO-5 (GF16 128×133)", "s": 1.69, "e_neon": 2088.5, "e_amx": 1510.4, "red": 27.7, "edp": 2.34, "win": True},
    {"name": "MAYO-3 (GF16 96×99)",   "s": 1.60, "e_neon": 857.1,  "e_amx": 653.4,  "red": 23.8, "edp": 2.10, "win": True},
    {"name": "MAYO-2 (GF16 64×78)",   "s": 1.53, "e_neon": 353.8,  "e_amx": 282.2,  "red": 20.2, "edp": 1.92, "win": True},
    {"name": "MAYO-1 (GF16 64×66)",   "s": 1.47, "e_neon": 251.6,  "e_amx": 207.5,  "red": 17.5, "edp": 1.79, "win": True},
    {"name": "sntrup761 (Ternary)",   "s": 1.52, "e_neon": 107.1,  "e_amx": 85.1,   "red": 20.5, "edp": 1.91, "win": True},
    {"name": "sntrup761 (Dense)",     "s": 0.92, "e_neon": 150.7,  "e_amx": 198.0,  "red": -31.3, "edp": 0.70, "win": False},
    {"name": "ML-KEM-512 PolyMul",    "s": 0.013, "e_neon": 2.5,   "e_amx": 234.5,  "red": -9132.4, "edp": 0.0001, "win": False},
    {"name": "ML-KEM-768 PolyMul",    "s": 0.013, "e_neon": 5.7,   "e_amx": 530.8,  "red": -9207.7, "edp": 0.0001, "win": False},
    {"name": "ML-KEM-1024 PolyMul",   "s": 0.012, "e_neon": 9.0,   "e_amx": 952.5,  "red": -10536.0, "edp": 0.0001, "win": False},
    {"name": "HQC-128 Sparse×Dense",  "s": 0.006, "e_neon": 21.0,  "e_amx": 4167.6, "red": -19736.3, "edp": 0.00003, "win": False},
    {"name": "HQC-128 Dense×Dense",   "s": 0.0006, "e_neon": 90.2, "e_amx": 168889.0, "red": -187055.4, "edp": 0.0000003, "win": False},
]

fig, ax = plt.subplots(figsize=(11.0, 6.4))
y_pos = np.arange(len(ROWS))[::-1]
bar_h = 0.32
gap = 0.05

# NEON and AMX colors from paper
COLOR_NEON = '#A7C7E7'
COLOR_AMX  = '#F2A7C3'
TEXT_NEON  = '#1565C0'
TEXT_AMX   = '#AD1457'

for i, (r, y) in enumerate(zip(ROWS, y_pos)):
    en = r["e_neon"]
    ea = r["e_amx"]
    win = r["win"]

    yn = y + bar_h/2 + gap/2
    ya = y - bar_h/2 - gap/2

    # NEON Bar
    bar_n = FancyBboxPatch((0.5, yn - bar_h/2), en - 0.5, bar_h,
                           boxstyle="round,pad=0,rounding_size=0.04",
                           facecolor=COLOR_NEON, edgecolor='none', label='NEON Baseline' if i == 0 else None)
    ax.add_patch(bar_n)

    # AMX Bar
    bar_a = FancyBboxPatch((0.5, ya - bar_h/2), ea - 0.5, bar_h,
                           boxstyle="round,pad=0,rounding_size=0.04",
                           facecolor=COLOR_AMX, edgecolor='none', label='AMX Accelerated' if i == 0 else None)
    ax.add_patch(bar_a)

    # NEON text label
    lbl_n = f"{en:.0f} µJ" if en >= 10 else f"{en:.1f} µJ"
    ax.text(en * 1.18, yn, lbl_n, va='center', ha='left', fontsize=8.0, color=TEXT_NEON, fontweight='medium')

    # AMX text label (put inside bar if very wide to prevent overlap)
    lbl_a = f"{ea/1000:.1f} mJ" if ea >= 1000 else f"{ea:.0f} µJ"
    if ea > 50000:  # e.g. HQC Dense (168.9 mJ)
        ax.text(ea * 0.72, ya, lbl_a, va='center', ha='right', fontsize=8.0, color='#FFFFFF', fontweight='bold')
    else:
        ax.text(ea * 1.18, ya, lbl_a, va='center', ha='left', fontsize=8.0, color=TEXT_AMX, fontweight='bold' if win else 'medium')

    # Right summary badge
    if win:
        badge_text = f"S = {r['s']:.2f}×  |  -{r['red']:.1f}% Energy  |  EDP: {r['edp']:.2f}×"
        box_bg = '#E8F5E9'
        box_edge = '#2E7D32'
        text_color = '#1B5E20'
    elif "Dense" in r["name"] and "sntrup" in r["name"]:
        badge_text = f"S = 0.92×  |  +31.3% Energy  |  EDP: 0.70×"
        box_bg = '#ECEFF1'
        box_edge = '#78909C'
        text_color = '#37474F'
    else:
        # Negative / mismatch cases
        if r['s'] < 0.001:
            badge_text = f"S = {r['s']:.4f}×  |  +{abs(r['red']):,.0f}% Energy"
        else:
            badge_text = f"S = {r['s']:.3f}×  |  +{abs(r['red']):,.0f}% Energy"
        box_bg = '#FFEBEE'
        box_edge = '#E57373'
        text_color = '#C62828'

    # Place badge at dedicated column on the right
    ax.text(7.5e5, y, badge_text, va='center', ha='left', fontsize=8.2, fontweight='bold', color=text_color,
            bbox=dict(boxstyle='round,pad=0.28', facecolor=box_bg, edgecolor=box_edge, lw=1.1))

# Light horizontal dividing lines between primitives for visual clarity
dividers = [6.5, 5.5, 1.5]
for div in dividers:
    ax.axhline(y=div, color='#ECEFF1', linestyle='-', linewidth=1.2, zorder=1)

# Annotations for algorithm groups on left or within the plot
ax.set_xscale('log')
ax.set_xlim(0.5, 6.0e6)
ax.set_ylim(-0.8, len(ROWS) - 0.2)
ax.set_yticks(y_pos)
ax.set_yticklabels([r["name"] for r in ROWS], fontsize=9.2, fontweight='medium', color='#2C3E50')
ax.set_xlabel('Isolated Dynamic Core Energy per Operation on Apple M3 (µJ, log scale)', fontweight='bold', color='#2C3E50', labelpad=9, fontsize=10)
ax.set_title('Cross-Scheme Active Energy per Operation on Apple M3: NEON Baseline vs. AMX Coprocessor',
             fontweight='bold', fontsize=11.5, color='#2C3E50', pad=14)

# Custom legend
ax.legend(loc='lower left', framealpha=0.96, facecolor='#FFFFFF', edgecolor='#CFD8DC', fontsize=9.0)

fig.tight_layout()

# Save outputs
fig.savefig("fig-energy-metrics.pdf", bbox_inches='tight')
fig.savefig("fig-energy-metrics.png", dpi=300, bbox_inches='tight')
fig.savefig("fig-energy-metrics.svg", bbox_inches='tight')
fig.savefig("single_bar_horizontal.png", dpi=300, bbox_inches='tight')

# Also mirror to desktop export folder
desktop_dir = "/Users/ca5/Desktop/PQC_AMX_Diagrams"
if os.path.exists(desktop_dir):
    shutil.copy("fig-energy-metrics.pdf", os.path.join(desktop_dir, "fig-energy-metrics.pdf"))
    shutil.copy("fig-energy-metrics.png", os.path.join(desktop_dir, "fig-energy-metrics.png"))
    shutil.copy("single_bar_horizontal.png", os.path.join(desktop_dir, "single_bar_horizontal.png"))

plt.close(fig)
print("Successfully generated single bar chart in PDF, PNG, SVG.")
