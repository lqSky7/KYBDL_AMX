#!/usr/bin/env python3
"""
AMX Conv2D Benchmark Plotting Script (Python / Matplotlib)
Generates publication-quality charts with customized palette and rounded cards/bars.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Create output directory
os.makedirs("figures", exist_ok=True)

# -------------------------------------------------------------------------
# Exact Color Palette from Specification
# -------------------------------------------------------------------------
COLORS = {
    "teal_6":    "#F0F7F7",
    "teal_50":   "#80BFBF",
    "teal_55":   "#73B9B9",
    "teal_60":   "#66B3B3",  # Apple AMX
    "blue_22":   "#C7C7FF",
    "blue_35":   "#A6A6FF",
    "blue_45":   "#8C8CFF",  # Scalar C
    "orange_25": "#FFDFBF",
    "orange_40": "#FFCC99",
    "orange_50": "#FFBF80",  # ARM NEON
    "red_5":     "#FFF2F2",
    "red_45":    "#FF8C8C",  # Highlight / 5x5
    "gray_12":   "#F0F0F0",  # Plot background
    "gray_30":   "#D9D9D9",  # Grid lines
    "gray_45":   "#C6C6C6",
    "gray_50":   "#BFBFBF",
    "gray_55":   "#B9B9B9",
    "gray_60":   "#B3B3B3",  # Ticks & Borders
}

IMPL_COLORS = {
    "Scalar C":  COLORS["blue_45"],
    "ARM NEON":  COLORS["orange_50"],
    "Apple AMX": COLORS["teal_60"],
}

# -------------------------------------------------------------------------
# Benchmark Data (Apple M3, N=30 Trials)
# -------------------------------------------------------------------------
resolutions = ["64×64", "128×128", "256×256"]

# 3x3 Kernel Latency (ms) & SD
lat_3x3 = {
    "Scalar C":  [0.6620, 2.7399, 10.7325],
    "ARM NEON":  [0.4845, 1.9540,  7.6851],
    "Apple AMX": [0.3847, 1.5383,  6.3002],
}
sd_3x3 = {
    "Scalar C":  [0.0205, 0.3729, 0.6853],
    "ARM NEON":  [0.0256, 0.1814, 0.5908],
    "Apple AMX": [0.0213, 0.1051, 0.5280],
}

# 3x3 Kernel Throughput (GOPS)
gops_3x3 = {
    "Scalar C":  [10.69, 10.33, 10.55],
    "ARM NEON":  [14.61, 14.49, 14.74],
    "Apple AMX": [18.40, 18.40, 17.98],
}

# 5x5 Kernel Latency (ms)
lat_5x5 = {
    "Scalar C":  [0.6578, 2.7602, 10.2887],
    "ARM NEON":  [0.7539, 3.0839, 12.4563],
    "Apple AMX": [0.9331, 3.7262, 14.7877],
}

# Speedup vs Baselines (3x3)
speedup_vs_c    = [1.72, 1.78, 1.70]
speedup_vs_neon = [1.26, 1.27, 1.22]


def apply_rounded_bars(ax, bars, color, corner_radius=0.03):
    """Replaces standard rectangular bars with rounded FancyBboxPatch bars."""
    for bar in bars:
        x = bar.get_x()
        y = bar.get_y()
        w = bar.get_width()
        h = bar.get_height()
        bar.remove()  # Remove standard bar
        
        # Add Fancy rounded box
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={corner_radius}",
            facecolor=color,
            edgecolor=COLORS["gray_55"],
            linewidth=0.8,
            zorder=3
        )
        ax.add_patch(box)


# =========================================================================
# FIGURE 1: Latency by Resolution (3×3 Kernel)
# =========================================================================
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
fig.patch.set_facecolor("white")
ax.set_facecolor(COLORS["teal_6"])

x_indices = np.arange(len(resolutions))
bar_width = 0.22

for i, (impl, color) in enumerate(IMPL_COLORS.items()):
    offset = (i - 1) * (bar_width + 0.04)
    x_pos = x_indices + offset
    
    # Draw bars
    bars = ax.bar(x_pos, lat_3x3[impl], width=bar_width, color=color, label=impl)
    apply_rounded_bars(ax, bars, color, corner_radius=0.03)
    
    # Error bars
    ax.errorbar(
        x_pos, lat_3x3[impl], yerr=sd_3x3[impl],
        fmt='none', ecolor=COLORS["gray_60"], elinewidth=1.2, capsize=3, capthick=1.2, zorder=4
    )

ax.set_yscale('log')
ax.set_ylabel("Latency (ms, log scale)", fontsize=11, fontweight="bold", color="#2C3E50")
ax.set_xlabel("Image Resolution", fontsize=11, fontweight="bold", color="#2C3E50")
ax.set_title("2D Convolution Latency — 3×3 Kernel (Apple M3)", fontsize=12, fontweight="bold", pad=12, color="#1A252F")
ax.set_xticks(x_indices)
ax.set_xticklabels(resolutions, fontsize=10, fontweight="bold", color="#2C3E50")
ax.set_yticks([0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0])
ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())

# Grid & Spines
ax.grid(axis="y", linestyle="--", alpha=0.7, color=COLORS["gray_30"], zorder=0)
for spine in ax.spines.values():
    spine.set_color(COLORS["gray_45"])
    spine.set_linewidth(1.0)

# Legend with rounded box
legend = ax.legend(
    loc="upper left", frameon=True, facecolor="white", edgecolor=COLORS["gray_45"],
    fontsize=9.5, framealpha=0.95
)
legend.get_frame().set_boxstyle("round,pad=0.4,rounding_size=0.05")

plt.tight_layout()
plt.savefig("figures/fig1_latency_3x3_py.pdf")
plt.savefig("figures/fig1_latency_3x3_py.png")
plt.close()
print("Saved: figures/fig1_latency_3x3_py.pdf/.png")


# =========================================================================
# FIGURE 2: Throughput (GOPS) — 3×3 Kernel
# =========================================================================
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
fig.patch.set_facecolor("white")
ax.set_facecolor(COLORS["teal_6"])

for i, (impl, color) in enumerate(IMPL_COLORS.items()):
    offset = (i - 1) * (bar_width + 0.04)
    x_pos = x_indices + offset
    
    bars = ax.bar(x_pos, gops_3x3[impl], width=bar_width, color=color, label=impl)
    apply_rounded_bars(ax, bars, color, corner_radius=0.03)
    
    # Value labels on top of bars
    for x, val in zip(x_pos, gops_3x3[impl]):
        ax.text(x, val + 0.4, f"{val:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#2C3E50")

ax.set_ylim(0, 23)
ax.set_ylabel("Throughput (GOPS)", fontsize=11, fontweight="bold", color="#2C3E50")
ax.set_xlabel("Image Resolution", fontsize=11, fontweight="bold", color="#2C3E50")
ax.set_title("2D Convolution Throughput — 3×3 Kernel (Apple M3)", fontsize=12, fontweight="bold", pad=12, color="#1A252F")
ax.set_xticks(x_indices)
ax.set_xticklabels(resolutions, fontsize=10, fontweight="bold", color="#2C3E50")

ax.grid(axis="y", linestyle="--", alpha=0.7, color=COLORS["gray_30"], zorder=0)
for spine in ax.spines.values():
    spine.set_color(COLORS["gray_45"])
    spine.set_linewidth(1.0)

legend = ax.legend(
    loc="upper right", frameon=True, facecolor="white", edgecolor=COLORS["gray_45"],
    fontsize=9.5, framealpha=0.95
)
legend.get_frame().set_boxstyle("round,pad=0.4,rounding_size=0.05")

plt.tight_layout()
plt.savefig("figures/fig2_throughput_3x3_py.pdf")
plt.savefig("figures/fig2_throughput_3x3_py.png")
plt.close()
print("Saved: figures/fig2_throughput_3x3_py.pdf/.png")


# =========================================================================
# FIGURE 3: Speedup Trend vs Resolution (3×3 Kernel)
# =========================================================================
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
fig.patch.set_facecolor("white")
ax.set_facecolor(COLORS["teal_6"])

# Line 1: vs Scalar C
ax.plot(
    x_indices, speedup_vs_c, marker='o', markersize=8, linewidth=2.4,
    color=COLORS["blue_45"], label="AMX vs. Scalar C", zorder=4
)
# Line 2: vs NEON
ax.plot(
    x_indices, speedup_vs_neon, marker='s', markersize=8, linewidth=2.4, linestyle="--",
    color=COLORS["orange_50"], label="AMX vs. ARM NEON", zorder=4
)

# Text labels for data points
for x, y in zip(x_indices, speedup_vs_c):
    ax.annotate(
        f"{y:.2f}×", (x, y), textcoords="offset points", xytext=(0, 9),
        ha="center", fontsize=9.5, fontweight="bold", color=COLORS["blue_45"],
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=COLORS["blue_35"], lw=0.8)
    )

for x, y in zip(x_indices, speedup_vs_neon):
    ax.annotate(
        f"{y:.2f}×", (x, y), textcoords="offset points", xytext=(0, 9),
        ha="center", fontsize=9.5, fontweight="bold", color=COLORS["orange_50"],
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=COLORS["orange_40"], lw=0.8)
    )

# Reference baseline = 1.0x
ax.axhline(1.0, color=COLORS["gray_50"], linestyle=":", linewidth=1.2, zorder=2)
ax.text(2.35, 1.03, "Baseline (1.0×)", color=COLORS["gray_55"], fontsize=8.5, fontstyle="italic")

ax.set_ylim(0.8, 2.1)
ax.set_ylabel("Speedup Multiplier (×)", fontsize=11, fontweight="bold", color="#2C3E50")
ax.set_xlabel("Image Resolution", fontsize=11, fontweight="bold", color="#2C3E50")
ax.set_title("AMX Speedup Scaling across Image Resolutions (3×3)", fontsize=12, fontweight="bold", pad=12, color="#1A252F")
ax.set_xticks(x_indices)
ax.set_xticklabels(resolutions, fontsize=10, fontweight="bold", color="#2C3E50")

ax.grid(axis="y", linestyle="--", alpha=0.7, color=COLORS["gray_30"], zorder=0)
for spine in ax.spines.values():
    spine.set_color(COLORS["gray_45"])
    spine.set_linewidth(1.0)

legend = ax.legend(
    loc="lower right", frameon=True, facecolor="white", edgecolor=COLORS["gray_45"],
    fontsize=9.5, framealpha=0.95
)
legend.get_frame().set_boxstyle("round,pad=0.4,rounding_size=0.05")

plt.tight_layout()
plt.savefig("figures/fig3_speedup_scaling_py.pdf")
plt.savefig("figures/fig3_speedup_scaling_py.png")
plt.close()
print("Saved: figures/fig3_speedup_scaling_py.pdf/.png")


# =========================================================================
# FIGURE 4: 3×3 vs 5×5 Kernel Comparison
# =========================================================================
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
fig.patch.set_facecolor("white")
ax.set_facecolor(COLORS["teal_6"])

# Grouped bars for 3x3 vs 5x5 AMX
amx_3x3_lat = lat_3x3["Apple AMX"]
amx_5x5_lat = lat_5x5["Apple AMX"]

b1 = ax.bar(x_indices - 0.15, amx_3x3_lat, width=0.28, color=COLORS["teal_60"], label="3×3 Kernel (Tiled AMX)")
b2 = ax.bar(x_indices + 0.15, amx_5x5_lat, width=0.28, color=COLORS["red_45"], label="5×5 Kernel (Tiled AMX)")

apply_rounded_bars(ax, b1, COLORS["teal_60"], corner_radius=0.03)
apply_rounded_bars(ax, b2, COLORS["red_45"], corner_radius=0.03)

for x, val in zip(x_indices - 0.15, amx_3x3_lat):
    ax.text(x, val * 1.15, f"{val:.2f}ms", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=COLORS["teal_60"])

for x, val in zip(x_indices + 0.15, amx_5x5_lat):
    ax.text(x, val * 1.15, f"{val:.2f}ms", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=COLORS["red_45"])

ax.set_yscale('log')
ax.set_ylim(0.1, 40)
ax.set_ylabel("AMX Latency (ms, log scale)", fontsize=11, fontweight="bold", color="#2C3E50")
ax.set_xlabel("Image Resolution", fontsize=11, fontweight="bold", color="#2C3E50")
ax.set_title("AMX Execution Latency — 3×3 vs. 5×5 Kernel", fontsize=12, fontweight="bold", pad=12, color="#1A252F")
ax.set_xticks(x_indices)
ax.set_xticklabels(resolutions, fontsize=10, fontweight="bold", color="#2C3E50")
ax.set_yticks([0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())

ax.grid(axis="y", linestyle="--", alpha=0.7, color=COLORS["gray_30"], zorder=0)
for spine in ax.spines.values():
    spine.set_color(COLORS["gray_45"])
    spine.set_linewidth(1.0)

legend = ax.legend(
    loc="upper left", frameon=True, facecolor="white", edgecolor=COLORS["gray_45"],
    fontsize=9.5, framealpha=0.95
)
legend.get_frame().set_boxstyle("round,pad=0.4,rounding_size=0.05")

plt.tight_layout()
plt.savefig("figures/fig4_kernel_scaling_py.pdf")
plt.savefig("figures/fig4_kernel_scaling_py.png")
plt.close()
print("Saved: figures/fig4_kernel_scaling_py.pdf/.png")

print("\n>>> All Python figures generated successfully with custom palette & rounded elements! <<<")
