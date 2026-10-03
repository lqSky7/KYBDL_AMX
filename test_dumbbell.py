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

fig, ax = plt.subplots(figsize=(10.5, 6.4))

y_pos = np.arange(len(ROWS))[::-1]  # Top to bottom

# Background highlight for AMX Winner rows vs Mismatch rows
ax.axhspan(y_pos[4] - 0.45, y_pos[0] + 0.45, color='#E8F5E9', alpha=0.5, zorder=0)
ax.axhspan(y_pos[-1] - 0.45, y_pos[5] + 0.45, color='#FFEBEE', alpha=0.35, zorder=0)

ax.text(1.2, y_pos[0] + 0.25, "AMX Energy-Efficient Tier (Up to 27.7% Energy Saved)",
        fontsize=8.5, fontweight='bold', color='#2E7D32')
ax.text(1.2, y_pos[5] + 0.25, "AMX Energy-Inefficient Tier (Instruction / Algorithmic Mismatch)",
        fontsize=8.5, fontweight='bold', color='#C62828')

for i, (r, y) in enumerate(zip(ROWS, y_pos)):
    en = r["e_neon"]
    ea = r["e_amx"]
    win = r["win"]

    # Connecting arrow/line
    line_col = '#2E7D32' if win else '#C62828'
    arrow_style = "-|>"
    lw = 2.0 if win else 1.4

    # Connect from NEON to AMX
    ax.annotate("", xy=(ea, y), xytext=(en, y),
                arrowprops=dict(arrowstyle=arrow_style, color=line_col, lw=lw,
                                mutation_scale=10, shrinkA=5, shrinkB=5),
                zorder=3)

    # Plot NEON point (Blue Circle)
    ax.scatter(en, y, color='#4A90E2', s=75, edgecolors='#1565C0', lw=1.5, zorder=4)

    # Plot AMX point (Pink Square)
    ax.scatter(ea, y, color='#EC407A', s=75, marker='s', edgecolors='#AD1457', lw=1.5, zorder=4)

    # Values directly on or near the points
    if win:
        ax.text(en * 1.18, y, f"{en:.0f} µJ", va='center', ha='left', fontsize=8.0, color='#1565C0')
        ax.text(ea * 0.85, y, f"{ea:.0f} µJ", va='center', ha='right', fontsize=8.0, color='#AD1457', fontweight='bold')
    elif "sntrup" in r["name"]:
        ax.text(en * 0.82, y, f"{en:.0f} µJ", va='center', ha='right', fontsize=8.0, color='#1565C0')
        ax.text(ea * 1.20, y, f"{ea:.0f} µJ", va='center', ha='left', fontsize=8.0, color='#AD1457')
    else:
        ax.text(en * 0.75, y, f"{en:.1f} µJ" if en < 10 else f"{en:.0f} µJ", va='center', ha='right', fontsize=7.8, color='#1565C0')
        ax.text(ea * 1.25, y, f"{ea/1000:.1f} mJ" if ea >= 1000 else f"{ea:.0f} µJ", va='center', ha='left', fontsize=7.8, color='#AD1457')

    # Summary Badge on right margin
    if win:
        badge_text = f"S = {r['s']:.2f}×  |  -{r['red']:.1f}% Energy  |  EDP: {r['edp']:.2f}×"
        box_bg = '#C8E6C9'
        box_edge = '#2E7D32'
        text_color = '#1B5E20'
    elif "Dense" in r["name"] and "sntrup" in r["name"]:
        badge_text = f"S = 0.92×  |  +31% Energy  |  EDP: 0.70×"
        box_bg = '#ECEFF1'
        box_edge = '#78909C'
        text_color = '#37474F'
    else:
        badge_text = f"S = {r['s']:.4f}×  |  +{abs(r['red']):,.0f}% Energy" if r['s'] < 0.01 else f"S = {r['s']:.3f}×  |  +{abs(r['red']):,.0f}% Energy"
        box_bg = '#FFCDD2'
        box_edge = '#EF5350'
        text_color = '#B71C1C'

    ax.text(3.5e5, y, badge_text, va='center', ha='left', fontsize=8.0, fontweight='bold', color=text_color,
            bbox=dict(boxstyle='round,pad=0.28', facecolor=box_bg, edgecolor=box_edge, lw=1.0))

ax.set_xscale('log')
ax.set_xlim(0.8, 2.5e6)
ax.set_ylim(-0.8, len(ROWS) - 0.2)
ax.set_yticks(y_pos)
ax.set_yticklabels([r["name"] for r in ROWS], fontsize=9.2, fontweight='medium', color='#2C3E50')

ax.set_xlabel('Active Energy per Operation on Apple M3 (µJ, log scale)   ← Leftward Shift = Energy Saved by AMX',
              fontweight='bold', color='#2C3E50', labelpad=8)
ax.set_title('Isolated Energy Consumption Shift: NEON Baseline → AMX Coprocessor Across All Four Schemes',
             fontweight='bold', fontsize=11.5, color='#2C3E50', pad=14)

# Custom legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4A90E2', markeredgecolor='#1565C0', markersize=8.5, label='NEON Baseline'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#EC407A', markeredgecolor='#AD1457', markersize=8.5, label='AMX Accelerated'),
    plt.Line2D([0], [0], color='#2E7D32', lw=2.2, label='AMX Wins (Energy Saved, Leftward Arrow)'),
    plt.Line2D([0], [0], color='#C62828', lw=1.6, label='AMX Penalty (Energy Waste, Rightward Arrow)'),
]
ax.legend(handles=legend_elements, loc='lower left', framealpha=0.95, facecolor='#FFFFFF', edgecolor='#CFD8DC', fontsize=8.5)

fig.tight_layout()
fig.savefig("candidate_dumbbell.png", dpi=300)
plt.close(fig)
print("Generated candidate_dumbbell.png")
