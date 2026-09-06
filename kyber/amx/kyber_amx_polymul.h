#ifndef KYBER_AMX_POLYMUL_H
#define KYBER_AMX_POLYMUL_H

#include <stdint.h>

/*
 * AMX TMVP polynomial multiplication for Kyber.
 *
 * Computes c(x) = a(x) * b(x) mod (x^256 + 1) mod q, where q=3329.
 *
 * All coefficients are int16_t in [0, q) on input.
 * Output coefficients are in [0, q) after Barrett reduction.
 *
 * This replaces the NTT-based multiplication from the NEON implementation
 * with a Toeplitz Matrix-Vector Product (TMVP) mapped to AMX's MAC16 hardware,
 * following the approach from "PQC-AMX: Accelerating Saber and FrodoKEM on the
 * Apple M1 and M3 SoCs" (Gazzoni et al., IEEE ARITH 2024).
 *
 * Key adaptation from Saber: Saber uses mod 2^16 (free overflow), while Kyber
 * uses mod q=3329 (requires explicit Barrett reduction after accumulation).
 * The MAC16 32-bit accumulators can hold 8 * 3328^2 ~ 88.6M < 2^32 without overflow.
 */

#define KYBER_AMX_N 256
#define KYBER_AMX_Q 3329
#define KYBER_AMX_N32 (KYBER_AMX_N / 32)  /* = 8 slices of 32 coefficients */

/*
 * AMX polynomial multiplication mod q mod (x^n+1).
 * Inputs a, b: polynomials with coefficients in [0, q).
 * Output res: polynomial with coefficients reduced to [0, q).
 */
void amx_kyber_poly_basemul(int16_t res[KYBER_AMX_N],
                            const int16_t a[KYBER_AMX_N],
                            const int16_t b[KYBER_AMX_N]);

/*
 * AMX matrix-vector multiply: res[i] = sum_j A[i][j] * s[j] for i in [0,k).
 * A is k x k matrix of polynomials, s is k-vector of polynomials.
 * Result coefficients are reduced mod q.
 *
 * This computes the full matrix-vector product using the TMVP approach,
 * replacing the NTT-based approach (polyvec_ntt + pointwise_mul + invntt).
 *
 * Parameters:
 *   res: output k-vector of polynomials, coefficients in [0, q)
 *   a:   k x k matrix of polynomials, coefficients in [0, q)
 *   s:   k-vector of polynomials, coefficients in [0, q)
 *   k:   module dimension (2 for Kyber-512, 3 for Kyber-768, 4 for Kyber-1024)
 */
void amx_kyber_matrix_vector_mul(int16_t *res,
                                 const int16_t *a,
                                 const int16_t *s,
                                 int k);

/*
 * AMX inner product: res = sum_j b[j] * s[j] for j in [0,k).
 * Used in decapsulation.
 */
void amx_kyber_inner_prod(int16_t res[KYBER_AMX_N],
                          const int16_t *b,
                          const int16_t *s,
                          int k);

/*
 * Barrett reduction of 32-bit accumulated values mod q=3329.
 * Reduces n int32_t values in-place, storing results as int16_t.
 *
 * Uses NEON SIMD for vectorized Barrett reduction:
 *   t = ((int64_t)v * a + (1<<25)) >> 26, where v = round(2^26 / q)
 *   result = a - t * q
 */
void kyber_barrett_reduce_vec(int16_t *res, const int32_t *a, int n);

/*
 * Polynomial addition: c[i] = (a[i] + b[i]) mod q
 */
void amx_kyber_poly_add(int16_t c[KYBER_AMX_N],
                        const int16_t a[KYBER_AMX_N],
                        const int16_t b[KYBER_AMX_N]);

/*
 * Polynomial subtraction: c[i] = (a[i] - b[i]) mod q
 */
void amx_kyber_poly_sub(int16_t c[KYBER_AMX_N],
                        const int16_t a[KYBER_AMX_N],
                        const int16_t b[KYBER_AMX_N]);

#endif /* KYBER_AMX_POLYMUL_H */
