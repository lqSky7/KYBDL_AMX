#!/usr/bin/env python3
"""
Generate modernized, publication-quality figures for PQC-AMX:
1. amx-register-layout (Outer-Product Matrix Accumulator, desaturated palette, no headers/badges)
2. methodology-blocks (Four-Stage Evaluation Methodology Diagram, white background publication theme)

Outputs: SVG, PDF, and high-res PNG (300 DPI).
"""

import os
import subprocess

OUT_DIR = "/Users/ca5/Projects/KYBDL_AMX"

def generate_amx_register_layout_svg():
    """
    AMX Outer-Product Matrix Accumulation Diagram:
    - Removed top title and hardware description text
    - Removed 'Up to 1,024 MACs / cycle' badge
    - Small left-aligned clarity note
    - Desaturated, muted scientific palette:
        * X register: muted cool slate (#EEF0F4, border #9AA5B8, text #334155)
        * Y register: muted warm stone (#F5F2EE, border #BDB1A7, text #44403C)
        * Z matrix: muted sage/slate-teal (#EDF3F2, border #8FAFA9, formula #3B635E, product #1E293B)
    - Retains exact mathematical outer-product content:
        Z00 += X0Y0, Z10 += X1Y0, ..., Zn0 += XnY0
        Z01 += X0Y1, Z11 += X1Y1, ..., Zn1 += XnY1
        ...
        Z0n += X0Yn, Z1n += X1Yn, ..., Znn += XnYn
    - NO mention of GEMM convolutions
    """
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 880 435" width="880" height="435">
  <defs>
    <style>
      .reg-header { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; font-weight: 600; font-size: 13px; }
      .reg-cell-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; font-weight: 700; font-size: 14px; }
      .z-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; font-weight: 600; font-size: 13px; fill: #3B635E; }
      .z-product { font-family: "SF Mono", Menlo, Monaco, Consolas, monospace; font-weight: 600; font-size: 12.5px; fill: #1E293B; }
      .dots-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; font-weight: 700; font-size: 18px; fill: #94A3B8; }
      .grid-label { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; font-weight: 600; font-size: 12.5px; fill: #475569; }
      .clarity-note { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; font-weight: 400; font-size: 8.5px; fill: #64748B; }
      .brace-label { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; font-weight: 600; font-size: 12px; fill: #64748B; }
    </style>

    <filter id="soft-shadow" x="-3%" y="-3%" width="106%" height="108%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="1.5" stdDeviation="2.5" flood-color="#0F172A" flood-opacity="0.04" />
    </filter>

    <marker id="arrow-down" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 1 2 L 7 5 L 1 8 z" fill="#94A3B8" />
    </marker>

    <marker id="arrow-right" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 1 2 L 7 5 L 1 8 z" fill="#A8A29E" />
    </marker>
  </defs>

  <!-- Background Canvas -->
  <rect width="880" height="435" fill="#FFFFFF" />

  <!-- ======================================================== -->
  <!-- TOP: X REGISTER / VECTOR                                 -->
  <!-- ======================================================== -->
  <g transform="translate(18, 0)">
    <!-- X Register Label centered over X register cells -->
    <text x="414" y="24" text-anchor="middle" class="reg-header" fill="#475569">X Register File (X0, X1, ..., Xn)</text>

    <!-- X0 Cell -->
    <rect x="180" y="34" width="112" height="42" rx="5" fill="#EEF0F4" stroke="#9AA5B8" stroke-width="1.3" filter="url(#soft-shadow)" />
    <text x="236" y="60" text-anchor="middle" class="reg-cell-text" fill="#334155">X0</text>

    <!-- X1 Cell -->
    <rect x="306" y="34" width="112" height="42" rx="5" fill="#EEF0F4" stroke="#9AA5B8" stroke-width="1.3" filter="url(#soft-shadow)" />
    <text x="362" y="60" text-anchor="middle" class="reg-cell-text" fill="#334155">X1</text>

    <!-- X Ellipsis Cell -->
    <rect x="432" y="34" width="90" height="42" rx="5" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.3" stroke-dasharray="4,4" />
    <text x="477" y="60" text-anchor="middle" class="dots-text">⋯</text>

    <!-- Xn Cell -->
    <rect x="536" y="34" width="112" height="42" rx="5" fill="#EEF0F4" stroke="#9AA5B8" stroke-width="1.3" filter="url(#soft-shadow)" />
    <text x="592" y="60" text-anchor="middle" class="reg-cell-text" fill="#334155">Xn</text>

    <!-- Downward Arrows into Columns -->
    <line x1="236" y1="80" x2="236" y2="95" stroke="#94A3B8" stroke-width="1.6" marker-end="url(#arrow-down)" />
    <line x1="362" y1="80" x2="362" y2="95" stroke="#94A3B8" stroke-width="1.6" marker-end="url(#arrow-down)" />
    <line x1="592" y1="80" x2="592" y2="95" stroke="#94A3B8" stroke-width="1.6" marker-end="url(#arrow-down)" />
  </g>

  <!-- ======================================================== -->
  <!-- LEFT: Y REGISTER / VECTOR                                -->
  <!-- ======================================================== -->
  <g transform="translate(18, 0)">
    <!-- Y Register Label (Rotated on the left) -->
    <text x="-231" y="36" text-anchor="middle" transform="rotate(-90)" class="reg-header" fill="#57534E">Y Register File (Y0, Y1, ..., Yn)</text>

    <!-- Y0 Cell -->
    <rect x="66" y="100" width="86" height="62" rx="5" fill="#F5F2EE" stroke="#BDB1A7" stroke-width="1.3" filter="url(#soft-shadow)" />
    <text x="109" y="137" text-anchor="middle" class="reg-cell-text" fill="#44403C">Y0</text>
    <line x1="154" y1="131" x2="173" y2="131" stroke="#A8A29E" stroke-width="1.6" marker-end="url(#arrow-right)" />

    <!-- Y1 Cell -->
    <rect x="66" y="172" width="86" height="62" rx="5" fill="#F5F2EE" stroke="#BDB1A7" stroke-width="1.3" filter="url(#soft-shadow)" />
    <text x="109" y="209" text-anchor="middle" class="reg-cell-text" fill="#44403C">Y1</text>
    <line x1="154" y1="203" x2="173" y2="203" stroke="#A8A29E" stroke-width="1.6" marker-end="url(#arrow-right)" />

    <!-- Y Ellipsis Cell -->
    <rect x="66" y="244" width="86" height="44" rx="5" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.3" stroke-dasharray="4,4" />
    <text x="109" y="272" text-anchor="middle" class="dots-text">⋮</text>

    <!-- Yn Cell -->
    <rect x="66" y="298" width="86" height="62" rx="5" fill="#F5F2EE" stroke="#BDB1A7" stroke-width="1.3" filter="url(#soft-shadow)" />
    <text x="109" y="335" text-anchor="middle" class="reg-cell-text" fill="#44403C">Yn</text>
    <line x1="154" y1="329" x2="173" y2="329" stroke="#A8A29E" stroke-width="1.6" marker-end="url(#arrow-right)" />
  </g>

  <!-- ======================================================== -->
  <!-- CENTER: Z ACCUMULATOR MATRIX                             -->
  <!-- ======================================================== -->
  <g transform="translate(18, 0)">
    <!-- Row 0 (Y0) -->
    <!-- Cell (0,0) -->
    <rect x="180" y="100" width="112" height="62" rx="5" fill="#EDF3F2" stroke="#8FAFA9" stroke-width="1.3" filter="url(#soft-shadow)" />
    <text x="236" y="125" text-anchor="middle" class="z-title">Z00 +=</text>
    <text x="236" y="146" text-anchor="middle" class="z-product">X0Y0</text>

    <!-- Cell (1,0) -->
    <rect x="306" y="100" width="112" height="62" rx="5" fill="#EDF3F2" stroke="#8FAFA9" stroke-width="1.3" filter="url(#soft-shadow)" />
    <text x="362" y="125" text-anchor="middle" class="z-title">Z10 +=</text>
    <text x="362" y="146" text-anchor="middle" class="z-product">X1Y0</text>

    <!-- Ellipsis (..., 0) -->
    <rect x="432" y="100" width="90" height="62" rx="5" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.3" stroke-dasharray="4,4" />
    <text x="477" y="137" text-anchor="middle" class="dots-text">⋯</text>

    <!-- Cell (n,0) -->
    <rect x="536" y="100" width="112" height="62" rx="5" fill="#EDF3F2" stroke="#8FAFA9" stroke-width="1.3" filter="url(#soft-shadow)" />
    <text x="592" y="125" text-anchor="middle" class="z-title">Zn0 +=</text>
    <text x="592" y="146" text-anchor="middle" class="z-product">XnY0</text>

    <!-- Row 1 (Y1) -->
    <!-- Cell (0,1) -->
    <rect x="180" y="172" width="112" height="62" rx="5" fill="#EDF3F2" stroke="#8FAFA9" stroke-width="1.3" filter="url(#soft-shadow)" />
    <text x="236" y="197" text-anchor="middle" class="z-title">Z01 +=</text>
    <text x="236" y="218" text-anchor="middle" class="z-product">X0Y1</text>

    <!-- Cell (1,1) -->
    <rect x="306" y="172" width="112" height="62" rx="5" fill="#EDF3F2" stroke="#8FAFA9" stroke-width="1.3" filter="url(#soft-shadow)" />
    <text x="362" y="197" text-anchor="middle" class="z-title">Z11 +=</text>
    <text x="362" y="218" text-anchor="middle" class="z-product">X1Y1</text>

    <!-- Ellipsis (..., 1) -->
    <rect x="432" y="172" width="90" height="62" rx="5" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.3" stroke-dasharray="4,4" />
    <text x="477" y="209" text-anchor="middle" class="dots-text">⋯</text>

    <!-- Cell (n,1) -->
    <rect x="536" y="172" width="112" height="62" rx="5" fill="#EDF3F2" stroke="#8FAFA9" stroke-width="1.3" filter="url(#soft-shadow)" />
    <text x="592" y="197" text-anchor="middle" class="z-title">Zn1 +=</text>
    <text x="592" y="218" text-anchor="middle" class="z-product">XnY1</text>

    <!-- Row 2: Ellipsis Row -->
    <rect x="180" y="244" width="112" height="44" rx="5" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.3" stroke-dasharray="4,4" />
    <text x="236" y="272" text-anchor="middle" class="dots-text">⋮</text>

    <rect x="306" y="244" width="112" height="44" rx="5" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.3" stroke-dasharray="4,4" />
    <text x="362" y="272" text-anchor="middle" class="dots-text">⋮</text>

    <rect x="432" y="244" width="90" height="44" rx="5" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.3" stroke-dasharray="4,4" />
    <text x="477" y="272" text-anchor="middle" class="dots-text">⋱</text>

    <rect x="536" y="244" width="112" height="44" rx="5" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.3" stroke-dasharray="4,4" />
    <text x="592" y="272" text-anchor="middle" class="dots-text">⋮</text>

    <!-- Row 3 (Yn) -->
    <!-- Cell (0,n) -->
    <rect x="180" y="298" width="112" height="62" rx="5" fill="#EDF3F2" stroke="#8FAFA9" stroke-width="1.3" filter="url(#soft-shadow)" />
    <text x="236" y="323" text-anchor="middle" class="z-title">Z0n +=</text>
    <text x="236" y="344" text-anchor="middle" class="z-product">X0Yn</text>

    <!-- Cell (1,n) -->
    <rect x="306" y="298" width="112" height="62" rx="5" fill="#EDF3F2" stroke="#8FAFA9" stroke-width="1.3" filter="url(#soft-shadow)" />
    <text x="362" y="323" text-anchor="middle" class="z-title">Z1n +=</text>
    <text x="362" y="344" text-anchor="middle" class="z-product">X1Yn</text>

    <!-- Ellipsis (..., n) -->
    <rect x="432" y="298" width="90" height="62" rx="5" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.3" stroke-dasharray="4,4" />
    <text x="477" y="335" text-anchor="middle" class="dots-text">⋯</text>

    <!-- Cell (n,n) -->
    <rect x="536" y="298" width="112" height="62" rx="5" fill="#EDF3F2" stroke="#8FAFA9" stroke-width="1.3" filter="url(#soft-shadow)" />
    <text x="592" y="323" text-anchor="middle" class="z-title">Znn +=</text>
    <text x="592" y="344" text-anchor="middle" class="z-product">XnYn</text>

    <!-- Right Brace: "32 rows" -->
    <path d="M 662 102 Q 674 102 674 122 L 674 220 Q 674 230 686 230 Q 674 230 674 240 L 674 340 Q 674 360 662 360" 
          fill="none" stroke="#94A3B8" stroke-width="1.6" />
    <text x="696" y="234" class="brace-label">32 rows</text>
  </g>

  <!-- ======================================================== -->
  <!-- BOTTOM: GRID LABEL & LEFT-ALIGNED CLARITY NOTE           -->
  <!-- ======================================================== -->
  <!-- Grid Label centered under Z matrix -->
  <text x="432" y="394" text-anchor="middle" class="grid-label">32 × 32 Z-accumulator grid</text>

  <!-- Small clarity note left-aligned -->
  <text x="84" y="420" text-anchor="start" class="clarity-note">(symbolic n lanes shown for clarity; hardware outer product is 32 × 32)</text>
</svg>
'''
    return svg


def generate_methodology_blocks_svg():
    """
    Four-Stage Evaluation Methodology Diagram
    Clean white background publication style:
    - Pure white background canvas
    - Crisp dark rectangular card outlines (#0F172A)
    - Clean dark headers: STAGE 1..4 in slate, bold titles in deep charcoal
    - Horizontal divider lines spanning across each card, aligned across all 4 stages
    - EXACTLY 1 point per stage with circular ring bullet markers:
      1) Get baseline NEON implementation data
      2) Identify the step that can be optimised
      3) Map computational bottleneck to AMX outer-product primitive
      4) Run the benchmarks for AMX vs NEON
    - NO mention of M3 anywhere!
    - Dark forward arrows connecting Stage 1 -> 2 -> 3 -> 4
    - Dashed feedback line below from Stage 4 back to Stage 2: 're-tune AMX mapping'
    """
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1020 250" width="1020" height="250">
  <defs>
    <style>
      .stage-tag { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; font-weight: 600; font-size: 11px; letter-spacing: 1.6px; fill: #475569; }
      .stage-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; font-weight: 700; font-size: 14px; fill: #0F172A; }
      .bullet-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; font-weight: 400; font-size: 12px; fill: #1E293B; }
      .feedback-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; font-weight: 500; font-size: 11.5px; fill: #334155; }
    </style>

    <marker id="fwd-arrow" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 1 1.5 L 8 5 L 1 8.5 z" fill="#0F172A" />
    </marker>

    <marker id="feedback-arrow-up" viewBox="0 0 10 10" refX="5" refY="3" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 1.5 8 L 5 1 L 8.5 8 z" fill="#334155" />
    </marker>
  </defs>

  <!-- White Background Canvas -->
  <rect width="1020" height="250" fill="#FFFFFF" />

  <!-- ======================================================== -->
  <!-- STAGE 1: Baseline Acquisition                            -->
  <!-- ======================================================== -->
  <g transform="translate(32, 22)">
    <rect width="216" height="146" fill="#FFFFFF" stroke="#0F172A" stroke-width="1.3" />
    
    <!-- Stage Header -->
    <text x="14" y="24" class="stage-tag">STAGE 1</text>
    <text x="14" y="47" class="stage-title">Baseline Acquisition</text>
    
    <!-- Full-width Divider Line -->
    <line x1="0" y1="68" x2="216" y2="68" stroke="#0F172A" stroke-width="1.3" />

    <!-- 1 Single Point with Ring Bullet -->
    <circle cx="21" cy="94" r="3" fill="none" stroke="#0F172A" stroke-width="1.4" />
    <text x="32" y="98" class="bullet-text">Get baseline NEON</text>
    <text x="32" y="116" class="bullet-text">implementation data</text>
  </g>

  <!-- Forward Arrow 1 -> 2 -->
  <line x1="248" y1="90" x2="280" y2="90" stroke="#0F172A" stroke-width="1.4" marker-end="url(#fwd-arrow)" />

  <!-- ======================================================== -->
  <!-- STAGE 2: AMX Reformulation                               -->
  <!-- ======================================================== -->
  <g transform="translate(282, 22)">
    <rect width="216" height="146" fill="#FFFFFF" stroke="#0F172A" stroke-width="1.3" />
    
    <!-- Stage Header -->
    <text x="14" y="24" class="stage-tag">STAGE 2</text>
    <text x="14" y="47" class="stage-title">AMX Reformulation</text>
    
    <!-- Full-width Divider Line -->
    <line x1="0" y1="68" x2="216" y2="68" stroke="#0F172A" stroke-width="1.3" />

    <!-- 1 Single Point with Ring Bullet -->
    <circle cx="21" cy="94" r="3" fill="none" stroke="#0F172A" stroke-width="1.4" />
    <text x="32" y="98" class="bullet-text">Identify the step that</text>
    <text x="32" y="116" class="bullet-text">can be optimised</text>
  </g>

  <!-- Forward Arrow 2 -> 3 -->
  <line x1="498" y1="90" x2="530" y2="90" stroke="#0F172A" stroke-width="1.4" marker-end="url(#fwd-arrow)" />

  <!-- ======================================================== -->
  <!-- STAGE 3: Integration & Batching Analysis                  -->
  <!-- ======================================================== -->
  <g transform="translate(532, 22)">
    <rect width="216" height="146" fill="#FFFFFF" stroke="#0F172A" stroke-width="1.3" />
    
    <!-- Stage Header -->
    <text x="14" y="22" class="stage-tag">STAGE 3</text>
    <text x="14" y="42" class="stage-title">Integration &amp;</text>
    <text x="14" y="58" class="stage-title">Batching Analysis</text>
    
    <!-- Full-width Divider Line -->
    <line x1="0" y1="68" x2="216" y2="68" stroke="#0F172A" stroke-width="1.3" />

    <!-- 1 Single Point with Ring Bullet -->
    <circle cx="21" cy="94" r="3" fill="none" stroke="#0F172A" stroke-width="1.4" />
    <text x="32" y="98" class="bullet-text">Map computational bottleneck</text>
    <text x="32" y="116" class="bullet-text">to AMX outer product</text>
  </g>

  <!-- Forward Arrow 3 -> 4 -->
  <line x1="748" y1="90" x2="780" y2="90" stroke="#0F172A" stroke-width="1.4" marker-end="url(#fwd-arrow)" />

  <!-- ======================================================== -->
  <!-- STAGE 4: Benchmarking & Comparison                       -->
  <!-- ======================================================== -->
  <g transform="translate(782, 22)">
    <rect width="216" height="146" fill="#FFFFFF" stroke="#0F172A" stroke-width="1.3" />
    
    <!-- Stage Header -->
    <text x="14" y="22" class="stage-tag">STAGE 4</text>
    <text x="14" y="42" class="stage-title">Benchmarking &amp;</text>
    <text x="14" y="58" class="stage-title">Comparison</text>
    
    <!-- Full-width Divider Line -->
    <line x1="0" y1="68" x2="216" y2="68" stroke="#0F172A" stroke-width="1.3" />

    <!-- 1 Single Point with Ring Bullet -->
    <circle cx="21" cy="94" r="3" fill="none" stroke="#0F172A" stroke-width="1.4" />
    <text x="32" y="98" class="bullet-text">Run the benchmarks for</text>
    <text x="32" y="116" class="bullet-text">AMX vs NEON</text>
  </g>

  <!-- ======================================================== -->
  <!-- FEEDBACK LOOP: Stage 4 -> Stage 2 (re-tune AMX mapping)   -->
  <!-- ======================================================== -->
  <!-- Dashed Line Return Path -->
  <path d="M 890 168 
           L 890 200 
           L 390 200 
           L 390 174" 
        fill="none" stroke="#334155" stroke-width="1.3" stroke-dasharray="5,4" 
        marker-end="url(#feedback-arrow-up)" />

  <!-- Feedback Label below dashed line -->
  <text x="640" y="220" text-anchor="middle" class="feedback-text">re-tune AMX mapping</text>
</svg>
'''
    return svg


def render_formats(svg_content, base_name):
    svg_path = os.path.join(OUT_DIR, f"{base_name}.svg")
    pdf_path = os.path.join(OUT_DIR, f"{base_name}.pdf")
    png_path = os.path.join(OUT_DIR, f"{base_name}.png")
    
    with open(svg_path, 'w') as f:
        f.write(svg_content)
    print(f"Saved: {svg_path}")

    # Convert to PDF via rsvg-convert
    subprocess.run(["/opt/homebrew/bin/rsvg-convert", "-f", "pdf", svg_path, "-o", pdf_path], check=True)
    print(f"Saved: {pdf_path}")

    # Convert to high-res PNG (zoom 3x = ~300 DPI)
    subprocess.run(["/opt/homebrew/bin/rsvg-convert", "-z", "3", "-f", "png", svg_path, "-o", png_path], check=True)
    print(f"Saved: {png_path}")


if __name__ == '__main__':
    # 1. Figure 1: AMX Outer-Product Matrix Accumulator
    amx_svg = generate_amx_register_layout_svg()
    render_formats(amx_svg, "amx-register-layout")
    render_formats(amx_svg, "amx_outer_product")

    # 2. Figure 2: Four-Stage Evaluation Methodology Pipeline
    meth_svg = generate_methodology_blocks_svg()
    render_formats(meth_svg, "methodology-blocks")
    render_formats(meth_svg, "fig_methodology_pipeline")

    print("\nAll figures generated successfully!")
