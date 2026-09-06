# =========================================================================
# Figures 3–6: R/ggplot2 plotting scripts for AMX Conv2D paper
# =========================================================================
# Requirements: install.packages(c("ggplot2", "dplyr", "scales", "patchwork"))
# Run: Rscript figures/fig3456_plots.R
# Output: PDF files in figures/ directory
# =========================================================================

library(ggplot2)
library(dplyr)
library(scales)

# -------------------------------------------------------------------------
# Global theming: consistent font, colors, sizing across all figures
# -------------------------------------------------------------------------
amx_palette <- c(
  "Scalar C"   = "#4E79A7",   # steel blue
  "ARM NEON"   = "#F28E2B",   # amber
  "Apple AMX"  = "#59A14F"    # green
)

amx_theme <- theme_bw(base_size = 11) +
  theme(
    text             = element_text(family = ""),
    plot.title       = element_text(face = "bold", size = 12, hjust = 0.5),
    legend.position  = "bottom",
    legend.title     = element_text(face = "bold", size = 10),
    legend.text      = element_text(size = 9),
    axis.title       = element_text(size = 10),
    axis.text        = element_text(size = 9),
    panel.grid.minor = element_blank()
  )

# =========================================================================
# Data: Table 2 — 3x3 kernel latency (mean ± SD, ms), 30 trials
# =========================================================================
df_latency_3x3 <- data.frame(
  Resolution     = factor(rep(c("64×64", "128×128", "256×256"), each = 3),
                          levels = c("64×64", "128×128", "256×256")),
  Implementation = factor(rep(c("Scalar C", "ARM NEON", "Apple AMX"), 3),
                          levels = c("Scalar C", "ARM NEON", "Apple AMX")),
  Latency_ms     = c(0.8906, 0.4929, 0.3686,
                     2.5719, 1.3876, 1.0542,
                     8.9769, 5.4376, 4.2161),
  SD             = c(0.0412, 0.0158, 0.0094,
                     0.0621, 0.0241, 0.0125,
                     0.2145, 0.0812, 0.0418)
)

# =========================================================================
# Data: Throughput (GOPS), 3x3 kernel
# =========================================================================
df_throughput_3x3 <- data.frame(
  Resolution     = factor(rep(c("64×64", "128×128", "256×256"), each = 3),
                          levels = c("64×64", "128×128", "256×256")),
  Implementation = factor(rep(c("Scalar C", "ARM NEON", "Apple AMX"), 3),
                          levels = c("Scalar C", "ARM NEON", "Apple AMX")),
  GOPS           = c(7.95, 14.36, 19.20,
                     11.01, 20.40, 26.86,
                     12.62, 20.83, 26.86)
)

# =========================================================================
# Data: AMX latency 3x3 vs 5x5
# =========================================================================
df_amx_kernel <- data.frame(
  Resolution  = factor(rep(c("64×64", "128×128", "256×256"), each = 2),
                       levels = c("64×64", "128×128", "256×256")),
  Kernel      = factor(rep(c("3×3", "5×5"), 3),
                       levels = c("3×3", "5×5")),
  AMX_ms      = c(0.3686, 21.3609,
                  1.0542, 80.4970,
                  4.2161, 318.5513)
)

# Scalar C reference lines for Figure 5
df_scalar_ref <- data.frame(
  Resolution = factor(c("64×64", "128×128", "256×256"),
                      levels = c("64×64", "128×128", "256×256")),
  Scalar_C   = c(0.6548, 2.6834, 10.6246)
)

# =========================================================================
# Data: Speedup vs resolution, 3x3 kernel
# =========================================================================
df_speedup <- data.frame(
  Resolution = factor(rep(c("64×64", "128×128", "256×256"), 2),
                      levels = c("64×64", "128×128", "256×256")),
  Baseline   = factor(rep(c("vs. Scalar C", "vs. ARM NEON"), each = 3),
                      levels = c("vs. Scalar C", "vs. ARM NEON")),
  Speedup    = c(2.42, 2.44, 2.13,
                 1.34, 1.32, 1.29)
)


# =========================================================================
# FIGURE 3: Latency by resolution × implementation, 3×3, log-scale
# =========================================================================
fig3 <- ggplot(df_latency_3x3,
               aes(x = Resolution, y = Latency_ms, fill = Implementation)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.65, color = "gray30", linewidth = 0.3) +
  geom_errorbar(
    aes(ymin = Latency_ms - SD, ymax = Latency_ms + SD),
    position = position_dodge(width = 0.75), width = 0.2, linewidth = 0.4
  ) +
  scale_y_log10(
    labels = label_number(accuracy = 0.01),
    breaks = c(0.1, 0.2, 0.5, 1, 2, 5, 10)
  ) +
  scale_fill_manual(values = amx_palette) +
  labs(
    title = "Convolution Latency — 3×3 Kernel (Apple M3, N=30 trials)",
    x     = "Image Resolution",
    y     = "Latency (ms, log scale)",
    fill  = "Implementation"
  ) +
  amx_theme

ggsave("figures/fig3_latency_3x3.pdf", fig3, width = 14, height = 9, units = "cm")
cat("Saved: figures/fig3_latency_3x3.pdf\n")


# =========================================================================
# FIGURE 4: Throughput (GOPS) by resolution × implementation, 3×3
# =========================================================================
fig4 <- ggplot(df_throughput_3x3,
               aes(x = Resolution, y = GOPS, fill = Implementation)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.65, color = "gray30", linewidth = 0.3) +
  geom_text(
    aes(label = sprintf("%.1f", GOPS)),
    position = position_dodge(width = 0.75),
    vjust = -0.4, size = 2.8, fontface = "bold"
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  scale_fill_manual(values = amx_palette) +
  labs(
    title = "Convolution Throughput — 3×3 Kernel (Apple M3)",
    x     = "Image Resolution",
    y     = "Throughput (GOPS)",
    fill  = "Implementation"
  ) +
  amx_theme

ggsave("figures/fig4_throughput_3x3.pdf", fig4, width = 14, height = 9, units = "cm")
cat("Saved: figures/fig4_throughput_3x3.pdf\n")


# =========================================================================
# FIGURE 5: AMX latency at 3×3 vs 5×5, log scale
#   Shows the AMX_STZ pipeline-flush regression, not a scaling trend.
#   Includes Scalar C reference lines.
# =========================================================================

# Kernel size palette: two shades of green (AMX color family)
kernel_palette <- c("3×3" = "#59A14F", "5×5" = "#B6452C")

fig5 <- ggplot(df_amx_kernel,
               aes(x = Resolution, y = AMX_ms, fill = Kernel)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.65, color = "gray30", linewidth = 0.3) +
  # Scalar C reference segments (one per resolution group)
  geom_point(
    data = df_scalar_ref,
    aes(x = Resolution, y = Scalar_C),
    inherit.aes = FALSE,
    shape = 4, size = 3, stroke = 1.2, color = "#4E79A7"
  ) +
  geom_text(
    data = df_scalar_ref,
    aes(x = Resolution, y = Scalar_C,
        label = paste0("Scalar C: ", sprintf("%.2f", Scalar_C), " ms")),
    inherit.aes = FALSE,
    hjust = -0.15, vjust = -0.5, size = 2.3, color = "#4E79A7", fontface = "italic"
  ) +
  scale_y_log10(
    labels = label_number(accuracy = 0.01),
    breaks = c(0.1, 0.5, 1, 5, 10, 50, 100, 500)
  ) +
  scale_fill_manual(values = kernel_palette) +
  labs(
    title    = "Apple AMX Latency — 3×3 vs. 5×5 Kernel (Apple M3)",
    subtitle = "5×5 regression reflects per-pixel AMX_STZ pipeline-flush overhead,\nnot fundamental kernel-size scaling (see Section 3.2).",
    x        = "Image Resolution",
    y        = "Apple AMX Latency (ms, log scale)",
    fill     = "Kernel Size"
  ) +
  amx_theme +
  theme(plot.subtitle = element_text(size = 8, face = "italic", color = "gray40"))

ggsave("figures/fig5_amx_3x3_vs_5x5.pdf", fig5, width = 14, height = 10, units = "cm")
cat("Saved: figures/fig5_amx_3x3_vs_5x5.pdf\n")


# =========================================================================
# FIGURE 6: Speedup vs resolution (line chart), 3×3 kernel only
# =========================================================================
speedup_palette <- c("vs. Scalar C" = "#4E79A7", "vs. ARM NEON" = "#F28E2B")

fig6 <- ggplot(df_speedup,
               aes(x = Resolution, y = Speedup, color = Baseline,
                   linetype = Baseline, group = Baseline)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 3) +
  geom_text(
    aes(label = sprintf("%.2f×", Speedup)),
    vjust = -1, size = 3, show.legend = FALSE, fontface = "bold"
  ) +
  # Reference line: no speedup
  geom_hline(yintercept = 1, linetype = "dotted", color = "gray50", linewidth = 0.5) +
  annotate("text", x = 3.35, y = 1.03, label = "no speedup",
           size = 2.5, color = "gray50", fontface = "italic") +
  # L1 cache annotation at 256x256
  annotate("text", x = 2.85, y = 2.00,
           label = "L1 cache\nexceeded\n(Sec. 3.3)",
           size = 2.2, color = "gray45", fontface = "italic", lineheight = 0.9) +
  annotate("segment", x = 2.9, xend = 3.0, y = 2.08, yend = 2.13,
           arrow = arrow(length = unit(0.15, "cm")), color = "gray45", linewidth = 0.4) +
  scale_y_continuous(
    limits = c(0.8, 2.8),
    breaks = seq(0.5, 3.0, by = 0.5)
  ) +
  scale_color_manual(values = speedup_palette) +
  scale_linetype_manual(values = c("vs. Scalar C" = "solid", "vs. ARM NEON" = "dashed")) +
  labs(
    title    = "AMX Speedup vs. Resolution — 3×3 Kernel (Apple M3)",
    x        = "Image Resolution",
    y        = "Speedup (×)",
    color    = "Baseline",
    linetype = "Baseline"
  ) +
  amx_theme

ggsave("figures/fig6_speedup_vs_resolution.pdf", fig6, width = 14, height = 9, units = "cm")
cat("Saved: figures/fig6_speedup_vs_resolution.pdf\n")

cat("\nAll 4 R/ggplot2 figures generated successfully.\n")
