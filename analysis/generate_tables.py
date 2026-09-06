#!/usr/bin/env python3
"""
Analysis script for PQC-AMX Kyber / ML-KEM benchmark results (Pure Python standard library).

Generates:
1. LaTeX formatted table matching Table I in Gazzoni et al. (IEEE ARITH 2024)
2. CSV summary of cycle counts and speedup ratios
3. Amortization-adjusted comparison table (accounting for NTT precomputation savings)
"""

import os
import sys
import glob
import csv

def parse_benchmark_results(results_dir):
    data = []
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' not found.", file=sys.stderr)
        return data

    for txt_file in glob.glob(os.path.join(results_dir, "*.txt")):
        filename = os.path.basename(txt_file)
        if filename.startswith("sample") or not filename.endswith(".txt"):
            continue

        parts = filename[:-4].split(":")
        if len(parts) != 5:
            continue

        scheme_op, pset, alloc, impl, variant = parts

        with open(txt_file, "r") as f:
            for line in f:
                line = line.strip()
                if ":" not in line:
                    continue
                op, cycles_str = line.split(":", 1)
                try:
                    cycles = int(cycles_str.strip())
                except ValueError:
                    continue

                data.append({
                    "Scheme": scheme_op,
                    "ParameterSet": pset,
                    "Alloc": alloc,
                    "Impl": impl,
                    "Variant": variant,
                    "Operation": op.strip(),
                    "Cycles": cycles
                })

    return data

def get_cycles(data, pset, variant, op):
    matches = [d["Cycles"] for d in data if d["ParameterSet"] == pset and d["Variant"] == variant and d["Operation"] == op]
    return matches[0] if matches else None

def generate_table1_latex(data):
    params = ["kyber512", "kyber768", "kyber1024"]
    ops = ["crypto_kem_keypair", "crypto_kem_enc", "crypto_kem_dec", "MatrixVectorMul"]
    op_labels = {
        "crypto_kem_keypair": "KeyGen",
        "crypto_kem_enc": "Encaps",
        "crypto_kem_dec": "Decaps",
        "MatrixVectorMul": "MatVecMul"
    }

    latex = []
    latex.append(r"\begin{table}[ht]")
    latex.append(r"\centering")
    latex.append(r"\caption{Cycle counts (in thousands) and speedup ratios for ML-KEM on Apple M3.}")
    latex.append(r"\label{tab:mlkem_m3}")
    latex.append(r"\begin{tabular}{l l r r r}")
    latex.append(r"\hline")
    latex.append(r"\textbf{Parameter Set} & \textbf{Operation} & \textbf{NEON Baseline} & \textbf{AMX TMVP (Ours)} & \textbf{Speedup} \\")
    latex.append(r"\hline")

    for pset in params:
        for op in ops:
            neon_val = get_cycles(data, pset, "neon", op)
            amx_val = get_cycles(data, pset, "amx", op)

            if neon_val is not None and amx_val is not None and amx_val > 0:
                speedup = neon_val / amx_val
                latex.append(f"{pset.upper()} & {op_labels[op]} & {neon_val/1000:.2f}k & {amx_val/1000:.2f}k & \\textbf{{{speedup:.2f}$\\times$}} \\\\")
            else:
                latex.append(f"{pset.upper()} & {op_labels[op]} & N/A & N/A & N/A \\\\")
        latex.append(r"\hline")

    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    return "\n".join(latex)

def generate_amortization_analysis(data):
    summary = []
    summary.append("================================================================================")
    summary.append("AMORTIZATION-ADJUSTED ANALYSIS: NEON NTT vs AMX TMVP for ML-KEM")
    summary.append("================================================================================")
    summary.append("Key Research Insight:")
    summary.append(" - NEON NTT amortizes cost by storing pre-NTT transformed A & s in keys.")
    summary.append(" - AMX TMVP operates directly on coefficient polynomials (no precomputation).")
    summary.append("--------------------------------------------------------------------------------")

    params = ["kyber512", "kyber768", "kyber1024"]
    for pset in params:
        summary.append(f"\n--- Parameter Set: {pset.upper()} ---")

        for op in ["crypto_kem_keypair", "crypto_kem_enc", "crypto_kem_dec", "MatrixVectorMul"]:
            neon_c = get_cycles(data, pset, "neon", op)
            amx_c = get_cycles(data, pset, "amx", op)

            if neon_c and amx_c:
                speedup = neon_c / amx_c
                summary.append(f"  {op:20s}: NEON = {neon_c:8d} cycles | AMX = {amx_c:8d} cycles | Speedup = {speedup:6.2f}x")

    return "\n".join(summary)

def main():
    cpu = sys.argv[1] if len(sys.argv) > 1 else "M3"
    results_dir = f"speed_results_{cpu}"

    data = parse_benchmark_results(results_dir)
    if not data:
        print(f"No benchmark data found in '{results_dir}'. Run run_benchmarks.sh first.", file=sys.stderr)
        return

    # Print LaTeX table
    latex_tbl = generate_table1_latex(data)
    print("--- LaTeX Table ---")
    print(latex_tbl)

    # Print Amortization Analysis
    analysis_str = generate_amortization_analysis(data)
    print("\n" + analysis_str)

    # Save to files
    with open(f"analysis_table_{cpu}.tex", "w") as f:
        f.write(latex_tbl)

    with open(f"analysis_summary_{cpu}.txt", "w") as f:
        f.write(analysis_str)

    with open(f"benchmark_results_{cpu}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Scheme", "ParameterSet", "Alloc", "Impl", "Variant", "Operation", "Cycles"])
        writer.writeheader()
        writer.writerows(data)

    print(f"\nSaved outputs to analysis_table_{cpu}.tex, analysis_summary_{cpu}.txt, and benchmark_results_{cpu}.csv.")

if __name__ == "__main__":
    main()
