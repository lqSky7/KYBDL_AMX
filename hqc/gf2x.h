/*
 * GF(2)[X] Polynomial Arithmetic — Common API
 *
 * All implementations (reference C, NEON PMULL, AMX) conform to this interface.
 * Polynomials are stored as packed bit arrays in uint64_t words:
 *   a[i] stores bits [64*i .. 64*i+63] of the polynomial.
 *   Bit j of word i represents coefficient of x^(64*i + j).
 *
 * The ring is F_2[X] / (X^n - 1) where n is prime (HQC parameter).
 */

#ifndef HQC_GF2X_H
#define HQC_GF2X_H

#include <stdint.h>
#include <stddef.h>
#include "params.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Bit-level accessors for packed polynomial representation
 * ======================================================================== */

/* Get bit at position pos in polynomial a */
static inline int gf2x_get_bit(const uint64_t *a, size_t pos) {
    return (a[pos / 64] >> (pos % 64)) & 1;
}

/* Set bit at position pos in polynomial a */
static inline void gf2x_set_bit(uint64_t *a, size_t pos) {
    a[pos / 64] |= (1ULL << (pos % 64));
}

/* Clear bit at position pos in polynomial a */
static inline void gf2x_clear_bit(uint64_t *a, size_t pos) {
    a[pos / 64] &= ~(1ULL << (pos % 64));
}

/* ========================================================================
 * Reference C Implementation (schoolbook, portable)
 * ======================================================================== */

/*
 * Dense × Dense polynomial multiplication (reference schoolbook)
 * c(x) = a(x) * b(x) mod (x^n - 1) over GF(2)
 *
 * @param c     Output polynomial, packed bit array [GF2X_WORDS(n) words]
 * @param a     Input polynomial a, packed bit array [GF2X_WORDS(n) words]
 * @param b     Input polynomial b, packed bit array [GF2X_WORDS(n) words]
 * @param n     Polynomial ring dimension
 */
void gf2x_mul_ref(uint64_t *c, const uint64_t *a, const uint64_t *b, size_t n);

/*
 * Sparse × Dense polynomial multiplication (reference)
 * c(x) = a_sparse(x) * b(x) mod (x^n - 1) over GF(2)
 *
 * The sparse polynomial a is represented by an array of positions where
 * its coefficients are 1.
 *
 * @param c         Output polynomial [GF2X_WORDS(n) words]
 * @param positions Array of bit positions where a has coefficient 1
 * @param weight    Number of nonzero coefficients in a (Hamming weight)
 * @param b         Dense input polynomial [GF2X_WORDS(n) words]
 * @param n         Polynomial ring dimension
 */
void gf2x_mul_sparse_ref(uint64_t *c, const uint32_t *positions, size_t weight,
                          const uint64_t *b, size_t n);

/* ========================================================================
 * NEON PMULL Implementation (carry-less multiply + Karatsuba)
 * ======================================================================== */

/*
 * Dense × Dense polynomial multiplication (NEON PMULL + Karatsuba)
 */
void gf2x_mul_neon(uint64_t *c, const uint64_t *a, const uint64_t *b, size_t n);

/*
 * Sparse × Dense polynomial multiplication (NEON 128-bit XOR)
 */
void gf2x_mul_sparse_neon(uint64_t *c, const uint32_t *positions, size_t weight,
                           const uint64_t *b, size_t n);

/* ========================================================================
 * AMX Implementation (integer-encoded outer products)
 * ======================================================================== */

/*
 * Dense × Dense polynomial multiplication (AMX integer schoolbook)
 */
void gf2x_mul_amx(uint64_t *c, const uint64_t *a, const uint64_t *b, size_t n);

/*
 * Sparse × Dense polynomial multiplication (AMX vectorized accumulate)
 */
void gf2x_mul_sparse_amx(uint64_t *c, const uint32_t *positions, size_t weight,
                           const uint64_t *b, size_t n);

/* ========================================================================
 * Utility functions (shared across implementations)
 * ======================================================================== */

/*
 * Reduce a 2n-bit product polynomial modulo (x^n - 1)
 * Folds the upper n-1 bits back into the lower n bits via XOR.
 *
 * @param c     Output reduced polynomial [GF2X_WORDS(n) words]
 * @param prod  Input product polynomial [GF2X_WORDS(2*n) words]
 * @param n     Ring dimension
 */
void gf2x_reduce_xn_minus_1(uint64_t *c, const uint64_t *prod, size_t n);

/*
 * XOR polynomial b (shifted left by `shift` bits) into accumulator acc.
 * acc ^= (b << shift)
 * Operates on the full-product buffer (length >= GF2X_WORDS(n + shift)).
 *
 * @param acc       Accumulator buffer
 * @param b         Source polynomial [GF2X_WORDS(n) words]
 * @param shift     Bit shift amount
 * @param n         Length of source polynomial in bits
 * @param acc_words Total words available in acc
 */
void gf2x_xor_shifted(uint64_t *acc, const uint64_t *b, size_t shift,
                       size_t n, size_t acc_words);

/*
 * XOR two polynomials: c = a ^ b
 */
void gf2x_add(uint64_t *c, const uint64_t *a, const uint64_t *b, size_t words);

/*
 * Zero out a polynomial buffer
 */
void gf2x_zero(uint64_t *a, size_t words);

/*
 * Copy polynomial: dst = src
 */
void gf2x_copy(uint64_t *dst, const uint64_t *src, size_t words);

/*
 * Compare two polynomials for equality
 * Returns 1 if equal, 0 otherwise.
 */
int gf2x_equal(const uint64_t *a, const uint64_t *b, size_t words);

#ifdef __cplusplus
}
#endif

#endif /* HQC_GF2X_H */
