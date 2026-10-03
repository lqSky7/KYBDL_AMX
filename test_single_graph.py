import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.ticker as ticker

# Style configuration
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.facecolor': '#FFFFFF',
    'figure.facecolor': '#FFFFFF',
    'savefig.facecolor': '#FFFFFF',
    'axes.edgecolor': '#CFD8DC',
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.color': '#CFD8DC',
})

# Data
DATA = [
    {"name": "MAYO-5", "cat": "win", "s": 1.69, "e_red": 27.7, "edp": 2.34, "t_neon": 1054.8, "t_amx": 624.1, "e_neon": 2088.5, "e_amx": 1510.4},
    {"name": "MAYO-3", "cat": "win", "s": 1.60, "e_red": 23.8, "edp": 2.10, "t_neon": 435.1, "t_amx": 272.2, "e_neon": 857.1, "e_amx": 653.4},
    {"name": "MAYO-2", "cat": "win", "s": 1.53, "e_red": 20.2, "edp": 1.92, "t_neon": 181.4, "t_amx": 118.6, "e_neon": 353.8, "e_amx": 282.2},
    {"name": "sntrup761 (Ternary)", "cat": "win", "s": 1.52, "e_red": 20.5, "edp": 1.91, "t_neon": 55.8, "t_amx": 36.7, "e_neon": 107.1, "e_amx": 85.1},
    {"name": "MAYO-1", "cat": "win", "s": 1.47, "e_red": 17.5, "edp": 1.79, "t_neon": 129.7, "t_amx": 87.9, "e_neon": 251.6, "e_amx": 207.5},
    {"name": "sntrup761 (Dense)", "cat": "loss", "s": 0.92, "e_red": -31.3, "edp": 0.70, "t_neon": 78.1, "t_amx": 84.6, "e_neon": 150.7, "e_amx": 198.0},
    {"name": "ML-KEM-512", "cat": "loss", "s": 0.013, "e_red": -91.3, "edp": 0.0001, "t_neon": 1.33, "t_amx": 99.8, "e_neon": 2.5, "e_amx": 234.5},
    {"name": "ML-KEM-768", "cat": "loss", "s": 0.013, "e_red": -92.1, "edp": 0.0001, "t_neon": 2.97, "t_amx": 224.9, "e_neon": 5.7, "e_amx": 530.8},
    {"name": "ML-KEM-1024", "cat": "loss", "s": 0.012, "e_red": -95.0, "edp": 0.0001, "t_neon": 4.64, "t_amx": 400.2, "e_neon": 9.0, "e_amx": 952.5},
    {"name": "HQC Sparse", "cat": "loss", "s": 0.006, "e_red": -95.0, "edp": 0.00003, "t_neon": 11.0, "t_amx": 1812.0, "e_neon": 21.0, "e_amx": 4167.6},
    {"name": "HQC Dense", "cat": "loss", "s": 0.0006, "e_red": -95.0, "edp": 0.0000003, "t_neon": 47.0, "t_amx": 73430.0, "e_neon": 90.2, "e_amx": 168889.0},
]

# CANDIDATE 1: Quadrant Map
fig1, ax = plt.subplots(figsize=(8.8, 5.4))
ax.axvline(1.0, color='#90A4AE', linestyle='--', lw=1.5, zorder=2)
ax.axhline(0.0, color='#90A4AE', linestyle='--', lw=1.5, zorder=2)

# Shading
ax.axhspan(0, 35, xmin=0.37, xmax=1.0, color='#E8F5E9', alpha=0.7, zorder=1)
ax.axhspan(-100, 0, xmin=0.0, xmax=0.37, color='#FFEBEE', alpha=0.55, zorder=1)

ax.text(1.35, 31.5, "AMX Performance-per-Watt Win Zone\n(Faster + Up to 27.7% Less Energy)",
        fontsize=9.2, fontweight='bold', color='#2E7D32', ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#C8E6C9', edgecolor='#2E7D32', lw=1.2))

ax.text(0.003, -75, "AMX Penalty Zone\n(Slower + 90x - 1870x More Energy)",
        fontsize=8.8, fontweight='bold', color='#C62828', ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFCDD2', edgecolor='#C62828', lw=1.2))

for d in DATA:
    is_win = d["cat"] == "win"
    c = '#7E57C2' if is_win else '#E57373'
    sz = 140 if is_win else 90
    ax.scatter(d["s"], d["e_red"], color=c, s=sz, edgecolors='#FFFFFF', lw=1.5, zorder=5)

    if is_win:
        # Offsets
        offset_y = 1.8 if "MAYO-5" in d["name"] else -3.8 if "MAYO-3" in d["name"] else 1.8 if "MAYO-2" in d["name"] else -4.2
        offset_x = 1.04
        if "sntrup" in d["name"]:
            offset_y = -3.8
            offset_x = 0.96
            ha = 'right'
        else:
            ha = 'left'
        ax.annotate(f"{d['name']}\nS={d['s']:.2f}x, -{d['e_red']:.1f}% E\n(EDP: {d['edp']:.2f}x)",
                    xy=(d["s"], d["e_red"]), xytext=(d["s"] * offset_x, d["e_red"] + offset_y),
                    fontsize=8.0, fontweight='bold', color='#2C3E50', ha=ha,
                    arrowprops=dict(arrowstyle="-", color='#B0BEC5', lw=0.8))
    elif "Dense" in d["name"] and "sntrup" in d["name"]:
        ax.annotate(f"{d['name']}\nS=0.92x, +31% E",
                    xy=(d["s"], d["e_red"]), xytext=(d["s"] * 1.05, d["e_red"] - 3.0),
                    fontsize=7.8, color='#546E7A')
    elif "ML-KEM-512" in d["name"]:
        ax.annotate("ML-KEM (All 3 sets)\nS ~ 0.013x, +92x E", xy=(d["s"], d["e_red"]), xytext=(d["s"] * 1.5, -60),
                    fontsize=8.0, color='#C62828', fontweight='bold',
                    arrowprops=dict(arrowstyle="->", color='#C62828', lw=1.0))
    elif "HQC Dense" in d["name"]:
        ax.annotate("HQC-128\nS ~ 0.0006x, +1870x E", xy=(d["s"], d["e_red"]), xytext=(d["s"] * 1.8, -60),
                    fontsize=8.0, color='#C62828', fontweight='bold',
                    arrowprops=dict(arrowstyle="->", color='#C62828', lw=1.0))

ax.set_xscale('log')
ax.set_xlim(0.0003, 2.5)
ax.set_ylim(-98, 38)
ax.set_xlabel('Kernel Speedup S = t_NEON / t_AMX (log scale)', fontweight='bold', color='#2C3E50', labelpad=6)
ax.set_ylabel('Active Energy Reduction (%) [Positive = Energy Saved]', fontweight='bold', color='#2C3E50', labelpad=6)
ax.set_title('Cross-Scheme Energy Efficiency & Speedup on Apple M3 (Performance-per-Watt)',
             fontweight='bold', fontsize=11.5, color='#2C3E50', pad=12)

ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:g}x'))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{int(y)}%'))

fig1.tight_layout()
fig1.savefig("candidate_quadrant.png", dpi=300)
plt.close(fig1)

print("Generated candidate_quadrant.png")
