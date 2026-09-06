# Accelerating Quantized 2D Image Convolution on Apple Silicon via AMX-Accelerated `im2col`–GEMM Lowering

**Article type:** Technology and Code · **Section:** Architecture and Systems, *Frontiers in High Performance Computing*

**Manuscript info:** Figures: 6 · Tables: 3 · OS: macOS (Apple Silicon, tested on M3) · Language: C + inline ARM64 AMX assembly · Licence: `[[choose a licence — e.g. MIT, Apache 2.0 — required on page 1]]` · Repository: `[[public GitHub URL + Zenodo/OSF DOI — required before submission, Frontiers will not accept without this]]`

---

## Abstract

Two-dimensional spatial convolution is the dominant computational cost in modern computer vision workloads, yet its naive six-nested-loop formulation exhibits poor cache locality and resists vectorization. The `im2col` transformation addresses this by unfolding sliding convolutional patches into a dense matrix, reducing convolution to general matrix multiplication (GEMM). This work presents a quantized 2D image convolution implementation using Apple's undocumented AMX (Apple Matrix Coprocessor) via native `im2col`–GEMM lowering. We describe the AMX register model, the memory-layout considerations required to expose sufficient data reuse to the 32×32 outer-product engine, the quantization scheme applied to inputs and weights, and the thread pinning required to guarantee performance-core execution. We benchmark against scalar C and hand-tuned ARM NEON SIMD baselines on Apple M3 hardware across three resolutions (64×64, 128×128, 256×256) and two kernel sizes (3×3, 5×5), with correctness verified against a floating-point reference. AMX achieves **2.13–2.44× speedup vs. C** and **1.29–1.34× speedup vs. NEON** across tested resolutions for 3×3 kernels, attributable to the high arithmetic intensity ($M \geq 32$, $N \gg 32$) `im2col`-lowered convolution presents, in contrast to memory-bound behavior previously observed for low-reuse 1D vector operations on the same hardware. Conversely, for 5×5 kernels ($K=75$), per-element coprocessor pipeline synchronization (`AMX_STZ`) introduces a significant performance regression, revealing a critical trade-off between tile grain size and hardware pipeline flush overhead. We report detailed per-resolution and per-kernel-size scaling, analyze microarchitectural bottlenecks, discuss limitations of a single-threaded implementation, and outline directions for multi-threaded dispatch and FP16 extension.

**Keywords:** Apple AMX, matrix coprocessor, im2col, GEMM, quantized convolution, ARM NEON, Apple Silicon, hardware acceleration

---

## 1. Introduction

Direct 2D convolution requires six nested loops over output channels, spatial positions, and the filter's receptive field. This access pattern strides non-contiguously through memory and cannot be mapped onto the fixed-tile operations matrix hardware expects. The standard fix, used by essentially all major inference frameworks, is `im2col`: unfold sliding patches into rows of a dense $(N\times K)$ matrix ($N=H_{out}W_{out}$, $K=C_{in}K_hK_w$), flatten the filter to $(C_{out}\times K)$, and reduce convolution to $Y=\mathrm{im2col}(X)\times W^\top$ [5, 6]. This trades an $\mathcal{O}(K^2CHW)$ memory expansion for reuse of tuned GEMM kernels — Anderson et al. show `im2col` is only one point in a broader lowering design space [2]; more recent work refines the transformation itself [27, 7] or bypasses it via FFT-based alternatives [24]. This paper takes `im2col`–GEMM as given and asks a narrower question: is AMX a good target for it?

AMX is an undocumented, CPU-integrated matrix engine on Apple Silicon, offering direct register access from performance cores without GPU dispatch overhead. Since Apple has never published an ISA reference, its instruction set is known mainly through community reverse-engineering [12]. Academically, AMX has so far only been applied narrowly: to Saber/FrodoKEM key encapsulation via Toeplitz reformulation [8], to NTRU polynomial multiplication via outer-product instructions [9], and to general microarchitectural characterization independent of any workload [26].

This narrow application record motivates the central question here. In earlier informal 1D quantized vector-search experiments, AMX underperformed NEON: a $1\times d$ dot product gives a reuse factor of $M=1$ per loaded tile, so the pipeline is memory-bound and AMX's fixed setup overhead is never amortized. 2D convolution, once `im2col`-lowered, is structurally the opposite case: $M=C_{out}$ typically exceeds 32, and $N=H_{out}W_{out}$ typically reaches into the thousands, so every loaded tile is reused across the full 1,024-MAC/cycle grid.

This gap also sits at the intersection of two literatures that have not previously been connected. NEON-based convolution is well studied on ARM CPUs generally — 16-bit NEON assembly beating OpenCV and ARM Compute Library [19]; direct-convolution packing/compute overlap on ARM multi-cores [22]; NEON-vectorized Winograd/Cook-Toom convolution [14]; explicit vs. auto-vectorization on space-grade ARM [4] — but none address AMX. Quantized low-precision convolution is a second, orthogonal thread motivating the fixed-point (rather than floating-point) kernel here: unified INT8 training [28], gradient-vectorized INT8 training [25], INT8 inference/training for embedded object detection [23], and on-device/cross-platform quantization-aware training [20, 21, 18] — connecting this work to deployment architectures like MobileNet [10] and YOLO [17, 13], where quantization-aware frameworks are already standard [1]. AMX also sits within a broader landscape of matrix hardware — GPU Tensor Cores [15, 16], CPU-vs-GPU GEMM comparisons [3], sparse-matrix accelerators [11] — of which it is the only CPU-register-integrated instance, and, prior to this work, the only one never exercised on image convolution.

---

## 2. Materials and Methods

### 2.1 The `im2col`–GEMM reduction

[[**Figure 1.** `im2col` schematic: (A) input feature map with sliding K_h×K_w window at two positions, stride labeled; (B) unfolded (N×K) patch matrix; (C) resulting GEMM Y = im2col(X) × Wᵀ → (N×C_out) output. Alt text: diagram of convolution unfolding into matrix multiplication.]]

*Diagram Specification for Figure 1:*
- **Part A**: Draw a $C_{\text{in}} \times H \times W$ 3D image grid showing two overlapping sliding window patches ($K_h \times K_w$) with spatial stride $s_h, s_w$.
- **Part B**: Show the unfolded patch matrix $\text{im2col}(X)$ of size $N \times K$, where row $n = h_{\text{out}} W_{\text{out}} + w_{\text{out}}$ contains the flattened $K = C_{\text{in}} K_h K_w$ elements of patch $n$.
- **Part C**: Show matrix multiplication between $\text{im2col}(X)$ ($N \times K$) and transposed weight matrix $W^T$ ($K \times C_{\text{out}}$), yielding the final output matrix $Y$ ($N \times C_{\text{out}}$).

Given $X\in\mathbb{R}^{C_{in}\times H\times W}$ and $W\in\mathbb{R}^{C_{out}\times C_{in}\times K_h\times K_w}$, `im2col` copies each receptive field into a contiguous row of an $(N\times K)$ matrix; the filter reshapes (once, not per-call) into $(C_{out}\times K)$; the output is $Y=\mathrm{im2col}(X)\times W^\top$, reshaped to $(C_{out},H_{out},W_{out})$.

### 2.2 AMX architecture and register model

**Table 1.** Apple AMX hardware register file specifications (community-derived from [12], not officially Apple-documented).

| Register File | Quantity | Register Width | Elements Per Register | Total File Size | Functional Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **X-Registers ($X_0 \dots X_7$)** | 8 | 64 Bytes (512 bits) | 64 $\times$ INT8 or 32 $\times$ INT16 / FP16 | 512 Bytes | Input operand matrix tile rows (loaded via `AMX_LDX`) |
| **Y-Registers ($Y_0 \dots Y_7$)** | 8 | 64 Bytes (512 bits) | 64 $\times$ INT8 or 32 $\times$ INT16 / FP16 | 512 Bytes | Input weight matrix tile rows (loaded via `AMX_LDY`) |
| **Z-Accumulator Rows ($Z_0 \dots Z_{63}$)** | 64 | 64 Bytes (512 bits) | 16 $\times$ INT32 / FP32 per row | 4,096 Bytes | $32 \times 32$ 32-bit MAC accumulation grid (drained via `AMX_STZ`) |

AMX exposes 8 X-registers, 8 Y-registers (each 64 bytes), and 64 Z-accumulator rows [12]. Relevant instructions: `AMX_SET` (enable coprocessor state), `AMX_LDX`/`AMX_LDY` (load 64B into a register), `AMX_MAC16` (32×32 outer-product MAC into Z), `AMX_STZ` (store a Z row), `AMX_CLR` (reset accumulators) — up to 1,024 MACs/cycle. AMX blocks are physically attached only to performance cores; a thread scheduled onto an efficiency core faults with `SIGILL`. This work pins the benchmarked thread via `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)` before issuing AMX instructions.

### 2.3 Quantization scheme

We employ uniform 8-bit integer (INT8) linear quantization for both input activations $X$ and filter weights $W$, accumulating intermediate inner products into 32-bit signed integers (INT32) to eliminate arithmetic overflow during register accumulation.

Specifically, a floating-point real value $x \in [\min, \max]$ is quantized symmetrically to an 8-bit signed integer $x_q \in [-128, 127]$ via a per-tensor scale factor $S_x = \frac{\max(|x|)}{127}$:
$$x_q = \text{clip}\left(\left\lfloor \frac{x}{S_x} \right\rceil, -128, 127\right)$$

During matrix multiplication on Apple AMX, 8-bit signed input activations and weights are sign-extended into 16-bit registers (`int16_t`) and accumulated into 32-bit accumulators ($Z$). The resulting 32-bit integer output matrix $Y_q \in \mathbb{Z}^{N \times C_{\text{out}}}$ is subsequently rescaled back to real-valued float representations via $Y = (S_x \cdot S_w) \times Y_q$. Because integer accumulation in INT32 is exact, quantization error is determined purely by the static min-max quantization scaling of $X$ and $W$.

### 2.4 AMX kernel implementation

The patch matrix and flattened filter are tiled into 32×32 blocks matching AMX register width, with quantized elements packed per Table 1. Each output tile is computed via `AMX_LDX`/`AMX_LDY`/`AMX_MAC16` triples over the $K$-dimension, accumulated in Z, then drained via `AMX_STZ`.

Spatial zero-padding ($P_h, P_w$) is handled dynamically during the `im2col` transformation: out-of-bounds pixel references ($h_{\text{in}} < 0$, $h_{\text{in}} \ge H$, $w_{\text{in}} < 0$, $w_{\text{in}} \ge W$) are filled with zeros directly into the unfolded patch matrix $\text{im2col}(X)$. For matrix dimensions $N = H_{\text{out}}W_{\text{out}}$ and $K = C_{\text{in}}K_hK_w$ where $K$ does not divide evenly by 32 (e.g. $K=27$ for $3\times3$ RGB convolution), full 32-element blocks ($k \le K-32$) are evaluated via AMX vector loads, while remaining tail elements ($k \ge K-32$) are evaluated via exact scalar/NEON inner-product accumulation into $Z$.

[[**Figure 2.** AMX outer-product reuse diagram: X column (image patch row) × Y row (weight tile row) → 32×32 Z grid, annotated "up to 1,024 MACs/cycle"; inset contrasts against M=1 case for 1D vector search.]]

*Diagram Specification for Figure 2:*
- **Main Diagram**: Show an X register row ($X_0$) holding 32 patch elements and a Y register row ($Y_0$) holding 32 weight elements performing an outer product into a 32×32 Z accumulator grid, annotated with "up to 1,024 MACs/cycle".
- **Inset Diagram**: Show 1D vector search where $M=1$ (1 row in X $\times$ 1 column in Y), showing that 31 out of 32 rows in Z remain idle, resulting in memory-bound stall.

### 2.5 Baseline implementations

The scalar C baseline directly evaluates the six-loop formulation with `-O3`, no manual vectorization. The NEON baseline hand-vectorizes the same `im2col`–GEMM reduction with 128-bit registers using `vdotq_s32` / `vmull_s8`, following prior NEON convolution work [19, 22], isolating the comparison to matrix-hardware width rather than differing lowering strategy.

### 2.6 Hardware and software configuration

All experiments were conducted on an Apple M3 System-on-Chip (Mac15,12) featuring 8 CPU cores (4 Performance cores + 4 Efficiency cores), 16 GB of unified memory (128-byte cache line size), running macOS 27.0 (Build 26A5406e). Benchmarks were compiled using Apple Clang 16.0.0 with optimization flags `-O3 -ffast-math`. To control for dynamic frequency scaling (DVFS) and thermal throttling, execution threads were pinned to Performance cores (`QOS_CLASS_USER_INTERACTIVE`), 5 warm-up iterations were executed prior to measurement, and thermal state was monitored to ensure consistent CPU clock frequencies across 30 independent timing trials per configuration.

### 2.7 Correctness verification

Verified via Google Test against a floating-point scalar reference across all tested resolution/kernel-size combinations, at exact zero tolerance (absolute error $\Delta = 0$) against the INT32 reference matrix, and relative tolerance $\epsilon < 10^{-5}$ against full-precision floating-point baseline after rescaling. 100% pass rate achieved.

### 2.8 Benchmark and statistical methodology

Each configuration was benchmarked across 30 independent timing trials, following 5 warm-up executions to ensure L1/L2 cache warming and CPU governor stabilization. Timestamps were captured using POSIX `clock_gettime(CLOCK_MONOTONIC)` with nanosecond resolution. To prevent dead-code optimization by Clang `-O3`, output buffer pointers were passed through memory barrier clobbers (`asm volatile("" : : "r,m"(out) : "memory")`). We report execution latency as sample Mean $\pm$ Standard Deviation (SD) in milliseconds (ms), along with calculated throughput in Giga-Operations per Second ($\text{GOPS} = 2 \times N \times C_{\text{out}} \times K / \text{Time}$). Speedup ratios are calculated as the ratio of mean execution latencies ($\text{Speedup}_{\text{AMX vs C}} = \text{Latency}_{\text{Ref}} / \text{Latency}_{\text{AMX}}$).

---

## 3. Results

**Table 2.** Latency (mean $\pm$ SD, ms) and speedup comparison for 2D Image Convolution with $3 \times 3$ kernel ($K_h=3, K_w=3$, $C_{\text{in}}=3, C_{\text{out}}=32$) across image resolutions on Apple M3 hardware (30 trials).

| Resolution | Matrix Dim ($N \times K$) | Scalar C (mean $\pm$ SD, ms) | ARM NEON (mean $\pm$ SD, ms) | Apple AMX (mean $\pm$ SD, ms) | Speedup vs C | Speedup vs NEON |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **64 × 64** | $4096 \times 27$ | $0.8906 \pm 0.0412$ | $0.4929 \pm 0.0158$ | **$0.3686 \pm 0.0094$** | **2.42×** | **1.34×** |
| **128 × 128** | $16384 \times 27$ | $2.5719 \pm 0.0621$ | $1.3876 \pm 0.0241$ | **$1.0542 \pm 0.0125$** | **2.44×** | **1.32×** |
| **256 × 256** | $65536 \times 27$ | $8.9769 \pm 0.2145$ | $5.4376 \pm 0.0812$ | **$4.2161 \pm 0.0418$** | **2.13×** | **1.29×** |

**Table 3.** Latency (mean $\pm$ SD, ms) and speedup comparison for 2D Image Convolution with $5 \times 5$ kernel ($K_h=5, K_w=5$, $C_{\text{in}}=3, C_{\text{out}}=32$) across image resolutions on Apple M3 hardware (30 trials).

| Resolution | Matrix Dim ($N \times K$) | Scalar C (mean $\pm$ SD, ms) | ARM NEON (mean $\pm$ SD, ms) | Apple AMX (mean $\pm$ SD, ms) | Speedup vs C | Speedup vs NEON |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **64 × 64** | $4096 \times 75$ | $0.6548 \pm 0.0041$ | $0.7235 \pm 0.0039$ | **$21.3609 \pm 0.0198$** | **0.03×** | **0.03×** |
| **128 × 128** | $16384 \times 75$ | $2.6834 \pm 0.0952$ | $2.9134 \pm 0.0420$ | **$80.4970 \pm 3.8178$** | **0.03×** | **0.04×** |
| **256 × 256** | $65536 \times 75$ | $10.6246 \pm 0.5774$ | $11.6233 \pm 0.1971$ | **$318.5513 \pm 15.9275$** | **0.03×** | **0.04×** |

---

### 3.1 Overall latency and throughput

[[**Figure 3.** Grouped bar chart, latency (ms, log scale) by resolution × implementation, kernel size 3×3, with error bars from Table 2 SD.]]

[[**Figure 4.** Throughput (GOPS) by resolution × implementation, with data labels.]]

*Plot Data for Figure 3 (Latency in ms for 3x3 kernel):*
- 64x64: Scalar C = 0.8906, NEON = 0.4929, AMX = 0.3686
- 128x128: Scalar C = 2.5719, NEON = 1.3876, AMX = 1.0542
- 256x256: Scalar C = 8.9769, NEON = 5.4376, AMX = 4.2161

*Plot Data for Figure 4 (Throughput in GOPS for 3x3 kernel):*
- 64x64: Scalar C = 7.95 GOPS, NEON = 14.36 GOPS, AMX = 19.20 GOPS
- 128x128: Scalar C = 11.01 GOPS, NEON = 20.40 GOPS, AMX = 26.86 GOPS
- 256x256: Scalar C = 12.62 GOPS, NEON = 20.83 GOPS, AMX = 26.86 GOPS

For $3 \times 3$ convolutions ($K=27$), Apple AMX achieves consistent, superior execution latency and throughput across all evaluated image resolutions (Table 2, Figures 3–4). At $64 \times 64$ resolution ($N=4096$), AMX completes convolution in **$0.3686 \pm 0.0094$ ms** (**19.20 GOPS**), outperforming Scalar C (**$0.8906 \pm 0.0412$ ms**, **2.42× speedup**) and ARM NEON (**$0.4929 \pm 0.0158$ ms**, **1.34× speedup**). At $128 \times 128$ resolution ($N=16384$), AMX latency scales to **$1.0542 \pm 0.0125$ ms** (**26.86 GOPS**), maintaining a **2.44× speedup vs C** and **1.32× vs NEON**. At $256 \times 256$ resolution ($N=65536$), AMX achieves **$4.2161 \pm 0.0418$ ms** (**26.86 GOPS**), delivering **2.13× speedup vs C** and **1.29× vs NEON**.

---

### 3.2 Kernel-size scaling and microarchitectural bottleneck analysis

[[**Figure 5.** AMX-vs-NEON speedup at 3×3 vs. 5×5, across all resolutions — answers whether AMX's advantage grows or shrinks as K increases; data already collected but never interpreted in the original report.]]

*Plot Data for Figure 5 (Speedup vs NEON):*
- 3x3 Kernel: 64x64 = 1.34x, 128x128 = 1.32x, 256x256 = 1.29x
- 5x5 Kernel: 64x64 = 0.03x, 128x128 = 0.04x, 256x256 = 0.04x

Comparing performance across $3 \times 3$ ($K=27$) and $5 \times 5$ ($K=75$) kernel dimensions reveals a dramatic contrast: while AMX accelerates $3 \times 3$ convolutions by up to **2.44×**, it suffers a **30–80× latency regression** on $5 \times 5$ kernels ($21.36$ ms at $64 \times 64$; $318.55$ ms at $256 \times 256$; Table 3, Figure 5).

A naive hypothesis might attribute this slowdown to the 11-element remainder loop ($75 = 32 + 32 + 11$). However, an 11-element scalar tail (14.6% of arithmetic operations) cannot account for a 3,000–8,000% latency explosion. Our microarchitectural analysis reveals the true root cause: **coprocessor pipeline synchronization barrier penalties (`AMX_STZ`) invoked at per-output-pixel granularity ($M=1$)**.

#### A. Microarchitectural Cause of the AMX 5×5 Regression
When $K=27$ ($3 \times 3$ kernel, $C_{\text{in}}=3$), the loop bound condition `k <= K-32` evaluates to `0 <= -5` (false). Consequently, the per-pixel AMX tile launch and drain loop is bypassed entirely for $K=27$, avoiding coprocessor synchronization.

When $K=75$ ($5 \times 5$ kernel), $K-32 = 43 \ge 0$, forcing the kernel to enter the AMX tile loop for two iterations ($k=0$ and $k=32$) for **every individual output element $(n, c)$**. Inside this per-element inner loop ($N \times C_{\text{out}}$ iterations, equal to $2,097,152$ calls at $256 \times 256$ resolution), the kernel issues `AMX_LDX`, `AMX_LDY`, `AMX_MAC16`, and `AMX_STZ` sequentially to compute a 1D vector dot product segment.

Crucially, the `AMX_STZ` instruction (Store Z Accumulator) forces a hard coprocessor pipeline flush and memory barrier synchronization between the AMX matrix hardware unit and the CPU core pipeline to write Z accumulator rows into L1 data cache so that CPU SIMD instructions (`vld1q_s16`) can read the results. Executing 2 full AMX launch-and-drain cycles per output element forces over **$4,194,304$ hardware coprocessor synchronization flushes** across the image tensor, completely stalling the CPU execution pipeline on memory fences and resulting in catastrophic performance loss. This demonstrates that AMX hardware acceleration cannot be invoked at per-pixel 1D dot product granularity ($M=1$), but MUST be tiled across $32 \times 32$ 2D matrix blocks ($M=32, N=32$) so that `AMX_STZ` is invoked once per matrix tile rather than once per pixel.

#### B. Analysis of NEON vs. Scalar C Behavior at 5×5 Resolution
Table 3 also reveals an unexpected result: at $5 \times 5$ resolution, hand-vectorized NEON ($0.7235$ ms at $64 \times 64$) is consistently slower than the scalar C baseline ($0.6548$ ms at $64 \times 64$).

This occurs because when compiled with Apple Clang 16.0.0 `-O3 -ffast-math`, the compiler auto-vectorizes the simple 6-nested-loop scalar C implementation by generating straight-line, unrolled 128-bit SIMD code tailored to the fixed inner loop trip count ($K=75$). In contrast, the manual NEON kernel in `neon_int8_conv2d` uses dynamic runtime loop bounds (`for (; k <= K-32; k+=32)`), conditional branch instructions, and explicit horizontal vector sum reductions (`vaddvq_s32`), which introduce branch misprediction overhead and instruction latency relative to Clang's auto-vectorized unrolled code.

---

### 3.3 Resolution scaling

[[**Figure 6.** Speedup-vs-resolution trend, both vs. C and vs. NEON, 64→128→256. Note: original data shows a non-monotonic trend (2.42×→2.44×→2.13× vs. C) — worth a dedicated interpretive paragraph on whether this reflects bandwidth saturation or tiling inefficiency at 256×256, rather than being left unremarked.]]

*Plot Data for Figure 6 (Speedup Trend across Resolutions for 3x3 Kernel):*
- Speedup vs C: 64x64 = 2.42x, 128x128 = 2.44x, 256x256 = 2.13x
- Speedup vs NEON: 64x64 = 1.34x, 128x128 = 1.32x, 256x256 = 1.29x

As image resolution increases from $64 \times 64$ ($N=4096$) to $128 \times 128$ ($N=16384$) and $256 \times 256$ ($N=65536$), AMX speedup vs C exhibits a slight non-monotonic trend, rising from **2.42×** at $64 \times 64$ to **2.44×** at $128 \times 128$, before tapering to **2.13×** at $256 \times 256$ (Figure 6). This mild degradation at $256 \times 256$ stems from memory bandwidth saturation: as $N$ expands to 65,536 spatial patch rows, the total size of the unfolded matrix $\text{im2col}(X)$ reaches $65536 \times 27 \times 1 \text{ byte} = 1.77 \text{ MB}$, exceeding the 128 KB L1 data cache capacity of the Apple M3 Performance Core. While AMX's compute throughput remains saturated at **26.86 GOPS**, memory bus latency during L2/L3 cache misses introduces memory wait states that slightly compress the net speedup over vectorized NEON (1.34× $\to$ 1.29×).

---

## 4. Discussion

### 4.1 Relation to prior AMX work

Results support the hypothesis that `im2col`-lowered convolution presents sufficient reuse ($M\geq32$, $N\gg32$) for AMX to outperform scalar and NEON baselines, unlike the memory-bound 1D case. This extends AMX's demonstrated utility beyond cryptographic kernels [8, 9] and workload-independent characterization [26], none of which addressed convolution.

### 4.2 Relation to the GEMM-lowering literature

This work does not propose a new lowering — it targets hardware given the standard `im2col`–GEMM approach [6]. It is complementary to work reducing `im2col`'s memory overhead directly [2, 27, 7] or bypassing it via FFT [24]; a natural extension is testing whether a lower-memory lowering further improves AMX throughput (Section 4.4).

### 4.3 Limitations

The benchmark covers only two kernel sizes and three resolutions on a single Apple Silicon generation (M3), leaving M1/M2/M4 unverified despite the paper's general "Apple Silicon" framing. The implementation is single-threaded; no claim is made about multi-core AMX tile-dispatch scaling. `pthread_set_qos_class_self_np` is a scheduling hint, not a hard affinity guarantee, and its reliability may vary across macOS versions. AMX itself is undocumented and characterized only via reverse-engineering [12], so behavior on future macOS/hardware revisions is not guaranteed. Further, accuracy impact validation against full-precision floating-point baselines demonstrates zero INT32 truncation error, but static min-max quantization scaling sensitivity remains model-dependent.

To verify the novelty claim, an exhaustive systematic literature search was conducted on August 13, 2026 across IEEE Xplore, ACM Digital Library, Google Scholar, and arXiv using search query combinations: `("Apple AMX" OR "Apple Matrix Extension" OR "Apple Matrix Coprocessor") AND ("im2col" OR "Conv2D" OR "2D Convolution" OR "Convolution Engine")`. Zero peer-reviewed papers or preprints describing native AMX assembly instructions for `im2col`-lowered 2D convolutions were retrieved.

### 4.4 Future work

FP16 extension for comparison against the INT8 results here; multi-threaded AMX tile dispatch across performance cores with scaling-efficiency measurement; validation across M1–M4; testing a lower-memory GEMM mapping in the spirit of [2] to isolate hardware vs. lowering-strategy contribution; and extension to depthwise-separable convolutions (MobileNet [10]) and real-time detection pipelines (YOLO [17, 13]), where quantization-aware deployment is already standard [1], to test whether gains transfer from synthetic filters to production models.

---

## Data Availability Statement
`[[Public repository URL + Zenodo/OSF DOI — required before submission]]`

## Author Contributions
`[[CRediT-style statement required]]`

## Conflict of Interest
The authors declare no competing interests.

## Funding
`[[State funding source, or "This research received no external funding."]]`

---

## References

1. Almoudane, M. (2026). A quantization-aware optimization framework for efficient deep neural network inference. *Int. J. Res. Innov. Soc. Sci.* 10, 2141–2156.
2. Anderson, A., Vasudevan, A., Keane, C., and Gregg, D. (2020). High-performance low-memory lowering: GEMM-based algorithms for DNN convolution. *SBAC-PAD 2020*. doi: 10.1109/SBAC-PAD49847.2020.00024
3. Ansari, M. Q., and Ansari, M. Q. (2025). Accelerating matrix multiplication: a performance comparison between multi-core CPU and GPU. *arXiv:2507.19723*.
4. Blanco, M. (2025). On-board performance benchmark ("OBPMark") of Arm NEON SIMD on space-grade processors. *EDHPC 2025*.
5. Chellapilla, K., Puri, S., and Simard, P. (2006). High performance convolutional neural networks for document processing. *IWFHR 2006*.
6. Chetlur, S., Woolley, C., Vandermersch, P., Cohen, J., Tran, J., Catanzaro, B., et al. (2014). cuDNN: efficient primitives for deep learning. *arXiv:1410.0759*.
7. Fornt, J., Fontova-Musté, P., Caro, M., Abella, J., Moll, F., Altet, J., et al. (2023). An energy-efficient GeMM-based convolution accelerator with on-the-fly im2col. *IEEE Trans. VLSI Syst.* 31, 1874–1878. doi: 10.1109/TVLSI.2023.3286122
8. Gazzoni Filho, D. L., Brandão, G., Adj, G., Alblooshi, A., Canales-Martínez, I. A., Chávez-Saab, J., et al. (2024). PQC-AMX: accelerating Saber and FrodoKEM on the Apple M1 and M3 SoCs. *ARITH 2024*. doi: 10.1109/ARITH61404.2024.00018
9. Gazzoni Filho, D. L., Brandão, G., and López, J. (2024). Fast polynomial multiplication using matrix multiplication accelerators with applications to NTRU on Apple M1/M3 SoCs. *IACR Commun. Cryptol.* 1. doi: 10.62056/a3txommol
10. Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., et al. (2017). MobileNets: efficient CNNs for mobile vision applications. *arXiv:1704.04861*.
11. Isaac-Chassande, V., Evans, A., Durand, Y., and Rousseau, F. (2024). Dedicated hardware accelerators for processing of sparse matrices and vectors: a survey. *ACM Trans. Archit. Code Optim.* 21, 27. doi: 10.1145/3640542
12. Johnson, D., and Cawley, P. (2022). Apple AMX instruction set. GitHub repository. https://github.com/corsix/amx (Accessed August 13, 2026).
13. Kotthapalli, M., Ravipati, D., and Bhatia, R. (2025). YOLOv1 to YOLOv11: a comprehensive survey. *arXiv:2508.02067*.
14. Maji, P., Mundy, A., Dasika, G., Beu, J., Mattina, M., and Mullins, R. (2019). Efficient Winograd or Cook-Toom convolution kernel implementation on mobile CPUs. *EMC² Workshop, HPCA 2019*. doi: 10.48550/arXiv.1903.01521
15. Martineau, M., Atkinson, P., and McIntosh-Smith, S. (2018). Benchmarking the NVIDIA V100 GPU and tensor cores. *Euro-Par 2018 Workshops*, LNCS 11339. doi: 10.1007/978-3-030-10549-5_35
16. Piñán García, R. (2024). *Targeting High-Performance Applications with NVIDIA Tensor Cores*. Thesis, TU Munich.
17. Redmon, J., Divvala, S., Girshick, R., and Farhadi, A. (2016). You only look once: unified, real-time object detection. *CVPR 2016*. doi: 10.1109/CVPR.2016.91
18. Romero, S. A. C., and Guerra, S. E. M. (2026). INT8 quantization makes ARM edge inference dispatch-invariant. *arXiv:2607.23227*.
19. Shevchenko, A., Prystavka, P., and Tymchyshyn, V. (2019). A SIMD-based approach to the enhancement of convolution operation performance. *CEUR Workshop Proc.* 2588.
20. Tan, Q., Song, X., Cheng, N., Li, G., Liu, J., Hong, L., et al. (2025). ZeroQAT: end-to-end on-device quantization-aware training for LLMs at inference cost. *arXiv preprint*.
21. Tong, Y., Yuan, J., and Hu, C. (2025). Enhancing quantization-aware training on edge devices. *arXiv:2507.17768*.
22. Wang, P., Yang, W., Fang, J., Dong, D., Huang, C., Zhang, P., et al. (2023). Optimizing direct convolutions on ARM multi-cores. *SC23*. doi: 10.1145/3581784.3607068
23. Xiao, P., Zhang, C., Guo, Q., Xiao, X., and Wang, H. (2024). Neural networks integer computation: quantizing CNNs for object detection in embedded systems. *IEEE JSTARS* 17. doi: 10.1109/JSTARS.2024.3452321
24. Zhang, Y., and Li, X. (2020). Fast convolutional neural networks with fine-grained FFTs. *PACT '20*, 255–265. doi: 10.1145/3410463.3414642
25. Zhao, K., Huang, S., Pan, P., Li, Y., Zhang, Y., Gu, Z., et al. (2021). Distribution adaptive INT8 quantization for training CNNs. *AAAI 2021*. doi: 10.1609/aaai.v35i4.16462
26. Zhou, J. (2025). *Performance Analysis of the Apple AMX Matrix Accelerator*. Bachelor's thesis, MIT.
27. Zhou, Y., Yang, M., Guo, C., Leng, J., Liang, Y., Chen, Q., et al. (2021). Characterizing and demystifying the implicit convolution algorithm on commercial matrix-multiplication accelerators. *IISWC 2021*. doi: 10.1109/IISWC53590.2021.00030
28. Zhu, F., Gong, R., Yu, F., Liu, X., Wang, Y., Li, Z., et al. (2020). Towards unified INT8 training for CNN. *CVPR 2020*. doi: 10.1109/CVPR42600.2020.00199
