/*
 * sntrup761 — NEON Polynomial Multiplication
 *
 * Uses vmlal_s16 (widening multiply-accumulate: int16 x int16 -> int32)
 * to compute schoolbook with 32-bit accumulators and no overflow.
 */

#include "poly.h"
#include <arm_neon.h>
#include <string.h>

void poly_mul_neon(int16_t c[SNTRUP_P], const int16_t a[SNTRUP_P],
                   const int16_t b[SNTRUP_P]) {
    /* Unreduced product: degree 2p-2 = 1520 */
    int32_t prod[2 * SNTRUP_P - 1];
    memset(prod, 0, sizeof(prod));

    /*
     * Schoolbook with NEON widening multiply-accumulate.
     * vmlal_s16: 4 x (int16 * int16 -> int32) accumulate per instruction.
     * We fix each coefficient of a and sweep through b in 4-wide NEON lanes.
     */
    for (int i = 0; i < SNTRUP_P; i++) {
        int16_t ai = a[i];
        if (ai == 0) continue;

        int16x4_t va = vdup_n_s16(ai);
        int32_t *dst = &prod[i];
        int j = 0;

        /* Process 4 coefficients at a time */
        for (; j + 3 < SNTRUP_P; j += 4) {
            int16x4_t vb = vld1_s16(&b[j]);
            int32x4_t vp = vld1q_s32(&dst[j]);
            vp = vmlal_s16(vp, va, vb);
            vst1q_s32(&dst[j], vp);
        }

        /* Remainder */
        for (; j < SNTRUP_P; j++) {
            dst[j] += (int32_t)ai * (int32_t)b[j];
        }
    }

    poly_reduce(c, prod);
}
