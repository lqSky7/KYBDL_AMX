#!/usr/bin/env python3

import os
import pandas as pd
import sys
import xlsxwriter
from xlsxwriter.utility import xl_rowcol_to_cell

sort_priority = {
    "640_AES": 0,
    "640_SHAKE": 1,
    "976_AES": 2,
    "976_SHAKE": 3,
    "1344_AES": 4,
    "1344_SHAKE": 5,
    "lightsaber": 6,
    "saber": 7,
    "firesaber": 8,
    "kyber512": 9,
    "kyber768": 10,
    "kyber1024": 11,
    "stack": 12,
    "mmap": 13,
    "ref": 14,
    "opt": 15,
    "neon": 16,
    "opt_amx": 17,
    "amx_polymul": 18,
    "amx_matmul": 19,
    "amx": 20,
    "": 21,
}


def my_sort(x):
    return sort_priority.get(x, 1337)


def my_sort_pd(series):
    return series.apply(lambda x: sort_priority.get(x, 1337))


def write_benchmarks(scheme, worksheet, df, columns):
    center_h = workbook.add_format({"align": "center"})
    center_h_bold = workbook.add_format({"align": "center", "bold": True})
    center_h_bold_2places = workbook.add_format(
        {"align": "center", "bold": True, "num_format": "0.00"}
    )
    center_hv = workbook.add_format({"align": "center", "valign": "vcenter"})

    match scheme:
        case "frodokem":
            benchmark_names = [
                ("Key generation", "crypto_kem_keypair"),
                ("Encapsulation", "crypto_kem_enc"),
                ("Decapsulation", "crypto_kem_dec"),
                ("Encapsulation 4x", "crypto_kem_enc4x"),
                ("Decapsulation 4x", "crypto_kem_dec4x"),
                ("A*s + e", "FrodoKEM A*s + e"),
                ("A*s + e matmul", "FrodoKEM A*s + e (matmul only)"),
                ("s*A + e", "FrodoKEM s*A + e"),
                ("s*A + e matmul", "FrodoKEM s*A + e (matmul only)"),
                ("s*A + e 4x", "FrodoKEM s*A + e 4x"),
                ("s*A + e matmul 4x", "FrodoKEM s*A + e 4x (matmul only)"),
            ]
        case "saber":
            benchmark_names = [
                ("Key generation", "crypto_kem_keypair"),
                ("Encapsulation", "crypto_kem_enc"),
                ("Decapsulation", "crypto_kem_dec"),
                ("MatrixVectorMulRound", "MatrixVectorMulRound"),
            ]
        case "kyber":
            benchmark_names = [
                ("Key generation", "crypto_kem_keypair"),
                ("Encapsulation", "crypto_kem_enc"),
                ("Decapsulation", "crypto_kem_dec"),
                ("MatrixVectorMul", "MatrixVectorMul"),
            ]

    impls = {}

    for pset in [
        "640_AES",
        "640_SHAKE",
        "976_AES",
        "976_SHAKE",
        "1344_AES",
        "1344_SHAKE",
    ]:
        impls[pset] = {
            "ref": {
                "ref": "Reference",
                "opt": "Optimized",
                "neon": "Ours (NEON)",
                "opt_amx": "Ours (AMX optimized)",
            }
        }

    for pset in ["lightsaber", "saber", "firesaber"]:
        impls[pset] = {
            "BHK": {
                "neon": "[BHK21]",
                "amx_polymul": "Ours (polynomial multiplication)",
                "amx_matmul": "Ours (matrix multiplication)",
            }
        }

    for pset in ["kyber512", "kyber768", "kyber1024"]:
        impls[pset] = {
            "": {
                "neon": "NEON baseline",
                "amx": "Ours (AMX TMVP)",
            }
        }

    for i in range(3):
        worksheet.merge_range(0, i, 1, i, columns[i + 1], center_hv)

    worksheet.merge_range(0, 3, 0, 3 + len(benchmark_names) - 1, "Operation", center_h)

    for i, names in enumerate(benchmark_names):
        worksheet.write(1, 3 + i, names[0], center_h)

    current_row = 2
    for pset in sorted(df["Parameter set"].unique(), key=my_sort):
        merge_pset_start = current_row

        df2 = df[df["Parameter set"] == pset]

        for memalloc in sorted(df2["Memory allocation"].unique(), key=my_sort):
            merge_memalloc_start = current_row

            df3 = df2[df2["Memory allocation"] == memalloc]

            for work in sorted(df3["Implementation"].unique(), key=my_sort):
                df4 = df3[df3["Implementation"] == work]

                for variant in sorted(df4["Variant"].unique(), key=my_sort):
                    impl = impls[pset][work][variant]
                    if not impl:
                        continue

                    worksheet.write(current_row, 2, impl, center_h)

                    df5 = df4[df4["Variant"] == variant]

                    for name, op in benchmark_names:
                        try:
                            cycle_count = int(
                                df5[df5["Operation"] == op]["Cycle count"].iloc[0]
                            )
                        except (IndexError, ValueError):
                            continue

                        col = [name[0] for name in benchmark_names].index(name)
                        worksheet.write(current_row, 3 + col, cycle_count, center_h)

                    current_row += 1

            worksheet.merge_range(
                merge_memalloc_start,
                1,
                current_row - 1,
                1,
                memalloc,
                center_hv,
            )

        worksheet.merge_range(
            merge_pset_start, 0, current_row - 1, 0, pset, center_hv
        )

    # Speedups
    merge_speedups_start = current_row
    for pset in sorted(df["Parameter set"].unique(), key=my_sort):
        merge_pset_start = current_row

        df2 = df[df["Parameter set"] == pset]

        for memalloc in sorted(df2["Memory allocation"].unique(), key=my_sort):
            merge_memalloc_start = current_row

            df3 = df2[df2["Memory allocation"] == memalloc]

            for work in sorted(df3["Implementation"].unique(), key=my_sort):
                df4 = df3[df3["Implementation"] == work]

                for variant in sorted(df4["Variant"].unique(), key=my_sort):
                    impl = impls[pset][work][variant]
                    if not impl:
                        continue

                    worksheet.write(
                        current_row,
                        2,
                        f"Speedup of {impl} over reference",
                        center_h_bold,
                    )

                    for name, op in benchmark_names:
                        col = [name[0] for name in benchmark_names].index(name)

                        # Excel formula for speedup calculation
                        target_cell = xl_rowcol_to_cell(current_row, 3 + col)
                        
                        cell_range_ref = f"{xl_rowcol_to_cell(2, 3 + col)}:{xl_rowcol_to_cell(current_row - 1, 3 + col)}"
                        
                        formula = f"=IFISERROR(MIN(IF(ISNUMBER(SEARCH(\")\", $C$3:$C${current_row - 1})), {cell_range_ref})) / {xl_rowcol_to_cell(current_row - 1, 3 + col)}, \"\")"
                        
                        worksheet.write_formula(
                            current_row, 3 + col, formula, center_h_bold_2places
                        )

                    current_row += 1


if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} CPU", file=sys.stderr)
    sys.exit(1)

cpu = sys.argv[1]
results_dir = f"speed_results_{cpu}"

if not os.path.exists(results_dir):
    print(f"Directory {results_dir} does not exist", file=sys.stderr)
    sys.exit(1)

data = []

for filename in os.listdir(results_dir):
    if filename.startswith("sample") or not filename.endswith(".txt"):
        continue

    parts = filename[:-4].split(":")
    if len(parts) != 5:
        continue

    scheme_op, pset, alloc, impl, variant = parts

    with open(os.path.join(results_dir, filename), "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            op_parts = line.split(":")
            if len(op_parts) != 2:
                continue
            op_name = op_parts[0].strip()
            try:
                cycles = int(op_parts[1].strip())
            except ValueError:
                continue

            data.append(
                {
                    "Scheme or operation": scheme_op,
                    "Parameter set": pset,
                    "Memory allocation": alloc,
                    "Implementation": impl,
                    "Variant": variant,
                    "Operation": op_name,
                    "Cycle count": cycles,
                }
            )

df = pd.DataFrame(data)

if df.empty:
    print("No benchmark data found.", file=sys.stderr)
    sys.exit(1)

excel_file = "benchmarks.xlsx"
writer = pd.ExcelWriter(excel_file, engine="xlsxwriter")
workbook = writer.book

columns = [
    "Scheme or operation",
    "Parameter set",
    "Memory allocation",
    "Implementation",
    "Variant",
    "Operation",
    "Cycle count",
]

for scheme in ["frodokem", "saber", "kyber"]:
    df_scheme = df[
        df["Parameter set"].isin(
            [
                "640_AES",
                "640_SHAKE",
                "976_AES",
                "976_SHAKE",
                "1344_AES",
                "1344_SHAKE",
            ]
            if scheme == "frodokem"
            else (
                ["lightsaber", "saber", "firesaber"]
                if scheme == "saber"
                else ["kyber512", "kyber768", "kyber1024"]
            )
        )
    ]
    if not df_scheme.empty:
        worksheet = workbook.add_worksheet(scheme)
        write_benchmarks(scheme, worksheet, df_scheme, columns)

writer.close()
print(f"Successfully generated {excel_file} with Kyber, Saber, and FrodoKEM benchmark results.")
