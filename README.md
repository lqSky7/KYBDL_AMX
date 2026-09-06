# PQC-AMX

Code for accelerating post-quantum crypto with the AMX matrix coprocessor on Apple Silicon (M1/M3).

Started with Saber and FrodoKEM. Now also covers ML-KEM (Kyber), MAYO, HQC, and SNTRUP Prime, plus two AMX demos: quantized 2D convolution and quantized vector search.

## Layout

- `amx/`: AMX helpers and Saber/FrodoKEM routines. Also `amx_vector_search.c`, the vector search engine.
- `amx_conv2d/`: quantized 2D image convolution on AMX (`im2col` + GEMM).
- `kyber/`: Kyber/ML-KEM AMX polymul plus NEON backends (512/768/1024).
- `mayo_uov/`: MAYO/UOV GF(16) AMX code.
- `hqc/`: HQC binary-polynomial (GF2x) ref, NEON, and AMX code.
- `sntrup761/`: SNTRUP-761 polynomial ref, NEON, and AMX code.
- `aes/`, `neon-ntt/`, `PQCrypto-LWEKE/`, `rng_opt/`: supporting crypto and RNG code, mostly from prior work (see original README history).
- `speed/`: benchmark programs. `wallclock.h` adds wall-clock timing next to the cycle counter.
- `test/`: Google Test checks for each scheme and engine.
- `scratch/`: small bring-up experiments (sources only; binaries are ignored).
- `analysis/`, `generate_figures.py`, `generate_jasp_dataset.py`: tables, plots, and datasets.
- `figures/`, `figures2/`, `paper_figures_fixed/`: figure sources and curated sets.
- `paper.tex`: the paper source.
- `run_benchmarks.sh`, `consolidate_benchmarks.py`: run and collect benchmarks.

## Requirements

- An Apple Silicon Mac (M1/M3 tested).
- CMake. Install with `brew install cmake`.
- Python 3 with `pandas` and `xlsxwriter` only for consolidating results.
- `sudo` access for benchmarks (cycle counter).

## Build

From the repo root:

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

Use Release. Debug can fail on the optimized `randombytes` routine due to a register allocation issue.

## Tests

From the `build` directory:

```
ctest
```

This runs all Google Test binaries, including KAT comparisons. You can also run a single test binary directly, e.g. `./test_kyber_kem`.

## Benchmarks

Binaries start with `speed_`. Run one directly, or run the full set:

```
sudo caffeinate ./run_benchmarks.sh
```

Run from the repo root. Results go to `speed_results_Mx` (x = your chip, e.g. M3). Close apps, turn off WiFi/Bluetooth, and plug in power for stable numbers.

To combine results into a spreadsheet:

```
python3 consolidate_benchmarks.py
```

## Paper and figures

- `paper.tex` is the paper source.
- `generate_figures.py` rebuilds the plots from benchmark CSVs.
- `analysis/generate_tables.py` rebuilds the LaTeX tables.

## License

Third-party code keeps its original license. Our changes to it use the same license. Our original code is CC0 1.0 (public domain). See `LICENSE` and per-folder headers.
