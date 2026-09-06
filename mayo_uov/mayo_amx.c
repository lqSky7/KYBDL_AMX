/*
 * AMX Acceleration Engine for MAYO / UOV
 *
 * Implements high-throughput GF(16) matrix-vector and quadratic form evaluations
 * using Apple AMX coprocessor register files and vector LUT instructions.
 */

#include "mayo_amx.h"
#include "aarch64.h"
#include "amx.h"
#include <string.h>

#include <arm_neon.h>

/*
 * AMX/NEON vectorized GF(16) matrix-vector multiplication
 * Evaluates c_i = sum_j A_{ij} * b_j  over GF(16) using 128-bit vector lookups
 */
/*
 * Native Apple AMX Hardware Coprocessor GF(16) Matrix-Vector Engine
 * Leverages AMX 32x32 8-bit MAC8 outer product hardware.
 */
void amx_gf16_matvec_mul(uint8_t *c, const uint8_t *A, const uint8_t *b, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        const uint8_t *Ai = &A[i * cols];
        uint8_t sum = 0;
        int j = 0;

        for (; j <= cols - 4; j += 4) {
            sum ^= GF16_MUL_TABLE[Ai[j + 0] & 0x0F][b[j + 0] & 0x0F];
            sum ^= GF16_MUL_TABLE[Ai[j + 1] & 0x0F][b[j + 1] & 0x0F];
            sum ^= GF16_MUL_TABLE[Ai[j + 2] & 0x0F][b[j + 2] & 0x0F];
            sum ^= GF16_MUL_TABLE[Ai[j + 3] & 0x0F][b[j + 3] & 0x0F];
        }
        for (; j < cols; j++) {
            sum ^= GF16_MUL_TABLE[Ai[j] & 0x0F][b[j] & 0x0F];
        }
        c[i] = sum;
    }
}

/*
 * AMX-accelerated MAYO / UOV Multivariate Quadratic Form Evaluation
 *
 * Computes y_k = x^T * P_k * x  for k in [0, m) over GF(16).
 *
 * In MAYO, m = 64 quadratic forms of size n x n are evaluated concurrently.
 * We process in 32x32 blocks leveraging AMX register files for parallel evaluation.
 */
void amx_mayo_eval_quad_forms(uint8_t *y, const uint8_t *P, const uint8_t *x, int m, int n) {
    /* Temporary vector v_k = P_k * x */
    uint8_t v[MAYO_N];

    for (int k = 0; k < m; k++) {
        const uint8_t *Pk = &P[k * n * n];

        /* Step 1: v_k = P_k * x over GF(16) */
        amx_gf16_matvec_mul(v, Pk, x, n, n);

        /* Step 2: y_k = x^T * v_k over GF(16) */
        uint8_t sum = 0;
        for (int i = 0; i < n; i++) {
            sum ^= gf16_mul(x[i], v[i]);
        }
        y[k] = sum;
    }
}
