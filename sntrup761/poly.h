/*
 * sntrup761 Polynomial Multiplication API
 *
 * All operations in Z_q[x] / (x^p - x - 1), q = 4591, p = 761.
 * Coefficients stored as int16_t in centered representation [-2295, 2295].
 */

#ifndef SNTRUP761_POLY_H
#define SNTRUP761_POLY_H

#include <stdint.h>
#include <stddef.h>
#include "params.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Reference C (schoolbook, 32-bit accumulator)
 * ======================================================================== */

/* Dense x Dense: c = a * b mod (x^p - x - 1) mod q */
void poly_mul_ref(int16_t c[SNTRUP_P], const int16_t a[SNTRUP_P],
                  const int16_t b[SNTRUP_P]);

/* ========================================================================
 * NEON (vmlal widening multiply-accumulate, 32-bit accumulator)
 * ======================================================================== */

void poly_mul_neon(int16_t c[SNTRUP_P], const int16_t a[SNTRUP_P],
                   const int16_t b[SNTRUP_P]);

/* ========================================================================
 * AMX (mac16 outer product with bounded accumulation)
 * ======================================================================== */

void poly_mul_amx(int16_t c[SNTRUP_P], const int16_t a[SNTRUP_P],
                  const int16_t b[SNTRUP_P]);

/* ========================================================================
 * Shared utilities
 * ======================================================================== */

/* Reduce x modulo q into centered range [-(q-1)/2, (q-1)/2] */
static inline int16_t mod_q(int32_t x) {
    /* Barrett-like reduction for q = 4591 */
    x = ((x % SNTRUP_Q) + SNTRUP_Q) % SNTRUP_Q;
    if (x > SNTRUP_Q_HALF) x -= SNTRUP_Q;
    return (int16_t)x;
}

/*
 * Reduce unreduced product polynomial mod (x^p - x - 1).
 * Input:  prod[0..2p-2] (int32_t, unreduced product of degree <= 2p-2)
 * Output: c[0..p-1] (int16_t, each reduced mod q)
 *
 * Reduction: x^p = x + 1, so x^{p+k} = x^{k+1} + x^k.
 * For k = 0..p-2: c[k] += prod[p+k], c[k+1] += prod[p+k]
 */
void poly_reduce(int16_t c[SNTRUP_P], const int32_t prod[2 * SNTRUP_P - 1]);

#ifdef __cplusplus
}
#endif

#endif /* SNTRUP761_POLY_H */
